"""
finetune/scripts/train_qlora.py
=================================
QLoRA fine-tuning pipeline for the Meesho support log parser.

Uses HuggingFace TRL's SFTTrainer with PEFT LoRA adapters and
bitsandbytes 4-bit quantization. Trains Qwen-2.5-7B-Instruct to
parse unstructured Meesho support logs into structured ticket JSON.

Requirements (install on GPU machine):
    pip install torch transformers trl peft bitsandbytes accelerate
    pip install datasets pyyaml

Usage:
    python finetune/scripts/train_qlora.py
    python finetune/scripts/train_qlora.py --config finetune/configs/qlora_qwen25_7b.yaml
    python finetune/scripts/train_qlora.py --config finetune/configs/qlora_qwen25_7b.yaml --resume

Note:
    This script requires a GPU with ≥16GB VRAM.
    On CPU it will fail at model load. Use RunPod, Lambda Labs,
    or Google Colab Pro for cloud GPU access.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(config: dict, resume_from_checkpoint: bool = False) -> None:
    """
    Run the full QLoRA fine-tuning pipeline.

    Parameters
    ----------
    config : dict
        Loaded YAML config.
    resume_from_checkpoint : bool
        If True, resume from the latest checkpoint in output_dir.
    """

    # ── Imports (deferred — these are GPU-only dependencies) ──────────
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    logger.info("Starting QLoRA training: %s", config["model_name"])
    logger.info("Output dir: %s", config["output_dir"])

    # ── 1. Load tokenizer ──────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=config.get("trust_remote_code", True),
        padding_side=config.get("padding_side", "right"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. BitsAndBytes 4-bit quantization config ──────────────────────
    logger.info("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(
            torch, config.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )

    # ── 3. Load base model ─────────────────────────────────────────────
    logger.info("Loading base model (this may take 3-5 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.get("trust_remote_code", True),
        torch_dtype=getattr(torch, config.get("torch_dtype", "bfloat16")),
        revision=config.get("model_revision", "main"),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.get("gradient_checkpointing", True),
    )

    # ── 4. LoRA adapter config ─────────────────────────────────────────
    logger.info("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=config.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Load dataset ────────────────────────────────────────────────
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_dataset"],
            "eval":  config["eval_dataset"],
        },
    )

    # ── 6. Format function — ChatML → single string ────────────────────
    def format_chatml(example: dict) -> dict:
        """
        Convert ChatML messages list to a single formatted string
        using Qwen's chat template.
        """
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chatml, num_proc=4)

    # ── 7. Training arguments ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        evaluation_strategy=config.get("evaluation_strategy", "steps"),
        eval_steps=config.get("eval_steps", 50),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 50),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        logging_steps=config.get("logging_steps", 10),
        logging_dir=config.get("logging_dir", "finetune/logs"),
        report_to=config.get("report_to", "none"),
        optim=config.get("optim", "paged_adamw_32bit"),
        bf16=True,
        dataloader_pin_memory=False,
    )

    # ── 8. SFTTrainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 1024),
        packing=config.get("packing", False),
        args=training_args,
    )

    # ── 9. Train ───────────────────────────────────────────────────────
    logger.info("Starting training...")
    checkpoint = None
    if resume_from_checkpoint:
        checkpoints = sorted(Path(config["output_dir"]).glob("checkpoint-*"))
        if checkpoints:
            checkpoint = str(checkpoints[-1])
            logger.info("Resuming from checkpoint: %s", checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)

    # ── 10. Save final adapter ─────────────────────────────────────────
    final_path = Path(config["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("✅ Training complete. Adapter saved to: %s", final_path)

    # Save training metrics
    metrics = trainer.evaluate()
    metrics_path = Path(config["output_dir"]) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Eval metrics: %s", metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Meesho RAG")
    parser.add_argument(
        "--config",
        default="finetune/configs/qlora_qwen25_7b.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_from_checkpoint=args.resume)
