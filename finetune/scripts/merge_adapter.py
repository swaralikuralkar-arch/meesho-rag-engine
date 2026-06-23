"""
finetune/scripts/merge_adapter.py
====================================
Merges the trained LoRA adapter weights back into the base model,
producing a single standalone model ready for deployment.

Why merge?
  During training, only the LoRA adapter weights are updated (~0.1% of
  total parameters). At inference time, running the adapter on top of
  the quantized base model adds latency. Merging produces a single
  full-precision model that runs at full speed without PEFT overhead.

  For deployment in the Meesho RAG pipeline, the merged model can be:
    - Served via vLLM for high-throughput inference
    - Quantized again with GGUF for CPU inference (llama.cpp)
    - Pushed to HuggingFace Hub for team sharing

Usage:
    python finetune/scripts/merge_adapter.py
    python finetune/scripts/merge_adapter.py \\
        --adapter finetune/checkpoints/qwen25_7b_meesho/final_adapter \\
        --output  finetune/merged_model/qwen25_7b_meesho_merged \\
        --push-to-hub  your-org/meesho-ticket-parser
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge(
    base_model_name: str,
    adapter_path: str | Path,
    output_path: str | Path,
    torch_dtype: str = "bfloat16",
    push_to_hub: str | None = None,
) -> None:
    """
    Merge LoRA adapter into base model and save.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model ID of the base model.
    adapter_path : str | Path
        Path to the saved LoRA adapter (from train_qlora.py).
    output_path : str | Path
        Where to save the merged model.
    torch_dtype : str
        Output dtype. "bfloat16" for most modern GPUs.
    push_to_hub : str | None
        If set, push merged model to this HuggingFace Hub repo.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    output_path  = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, torch_dtype)

    # ── 1. Load base model in full precision ───────────────────────────
    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 2. Load tokenizer ──────────────────────────────────────────────
    logger.info("Loading tokenizer from adapter path...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    # ── 3. Load and attach LoRA adapter ───────────────────────────────
    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # ── 4. Merge adapter weights into base model ───────────────────────
    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()
    logger.info("Merge complete.")

    # ── 5. Save merged model ───────────────────────────────────────────
    logger.info("Saving merged model to: %s", output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    logger.info("✅ Merged model saved.")

    # ── 6. Push to HuggingFace Hub (optional) ─────────────────────────
    if push_to_hub:
        logger.info("Pushing to HuggingFace Hub: %s", push_to_hub)
        model.push_to_hub(push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(push_to_hub)
        logger.info("✅ Model pushed to Hub: %s", push_to_hub)


# ---------------------------------------------------------------------------
# Quick inference test
# ---------------------------------------------------------------------------

def test_merged_model(model_path: str | Path) -> None:
    """
    Run a quick smoke test on the merged model to verify it works.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    logger.info("Running inference test on merged model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    test_log = (
        "supplier 84721 disputing rto deduction of 850rs on order MO-2847291. "
        "claims item was never returned. requesting reversal. marked urgent by agent Priya."
    )

    from finetune.scripts.prepare_dataset import SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": test_log},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
    )

    result = pipe(messages)
    output = result[0]["generated_text"][-1]["content"]

    print("\n── Inference Test ──────────────────────────────────")
    print(f"Input: {test_log}")
    print(f"Output:\n{output}")
    print("────────────────────────────────────────────────────\n")

    import json
    try:
        parsed = json.loads(output)
        assert parsed["supplier_id"] == "84721"
        assert parsed["issue_type"] == "RTO_DEDUCTION_DISPUTE"
        assert parsed["amount_disputed"] == 850.0
        print("✅ Inference test passed — JSON parsed correctly")
    except Exception as exc:
        print(f"⚠️  Inference test warning: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model HuggingFace ID",
    )
    parser.add_argument(
        "--adapter",
        default="finetune/checkpoints/qwen25_7b_meesho/final_adapter",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--output",
        default="finetune/merged_model/qwen25_7b_meesho_merged",
        help="Output path for merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="HuggingFace Hub repo to push merged model to",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inference test after merging",
    )
    args = parser.parse_args()

    merge(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output,
        torch_dtype=args.dtype,
        push_to_hub=args.push_to_hub,
    )

    if args.test:
        test_merged_model(args.output)
