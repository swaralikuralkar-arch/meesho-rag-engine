"""
finetune/eval_finetuned.py
============================
Evaluates the fine-tuned model against the Phase 3 golden dataset
and compares its scores against the baseline (gpt-4o-mini).

This is the final validation step before deploying the fine-tuned
model. It runs the same faithfulness gate used in CI but targets
the specific task the fine-tuned model was trained on: parsing
unstructured support logs into structured ticket JSON.

Usage:
    python finetune/eval_finetuned.py
    python finetune/eval_finetuned.py --model finetune/merged_model/qwen25_7b_meesho_merged
    python finetune/eval_finetuned.py --compare  # compare vs baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ticket parser evaluator
# ---------------------------------------------------------------------------

class TicketParserEvaluator:
    """
    Evaluates how accurately a model parses support logs into ticket JSON.

    Metrics:
      - field_accuracy   : fraction of fields correctly extracted
      - json_parse_rate  : fraction of outputs that are valid JSON
      - issue_type_acc   : accuracy on the issue_type enum field
      - amount_accuracy  : accuracy on currency amount extraction
        (within 1% tolerance for rounding)
    """

    REQUIRED_FIELDS = [
        "supplier_id", "issue_type", "order_id",
        "amount_disputed", "priority", "resolution_requested",
    ]

    def evaluate_output(
        self,
        predicted: str,
        ground_truth: dict,
    ) -> dict:
        """
        Score a single predicted JSON string against the ground truth.

        Returns a dict with per-field scores and overall accuracy.
        """
        # Try to parse JSON
        try:
            predicted_dict = json.loads(predicted)
            json_valid = True
        except (json.JSONDecodeError, TypeError):
            return {
                "json_valid": False,
                "field_accuracy": 0.0,
                "issue_type_correct": False,
                "amount_correct": False,
                "overall_score": 0.0,
            }

        correct_fields = 0
        total_fields = len(self.REQUIRED_FIELDS)

        for field in self.REQUIRED_FIELDS:
            pred_val = predicted_dict.get(field)
            true_val = ground_truth.get(field)

            if field == "amount_disputed":
                # Allow 1% tolerance for currency rounding
                if pred_val is not None and true_val is not None:
                    try:
                        if abs(float(pred_val) - float(true_val)) / max(float(true_val), 1) <= 0.01:
                            correct_fields += 1
                    except (ValueError, TypeError):
                        pass
            elif pred_val == true_val:
                correct_fields += 1

        field_accuracy = correct_fields / total_fields

        return {
            "json_valid": True,
            "field_accuracy": field_accuracy,
            "issue_type_correct": predicted_dict.get("issue_type") == ground_truth.get("issue_type"),
            "amount_correct": self._amount_correct(
                predicted_dict.get("amount_disputed"),
                ground_truth.get("amount_disputed"),
            ),
            "overall_score": field_accuracy,
        }

    @staticmethod
    def _amount_correct(pred, true) -> bool:
        if pred is None or true is None:
            return pred == true
        try:
            return abs(float(pred) - float(true)) / max(float(true), 1) <= 0.01
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Baseline comparison (using OpenAI API)
# ---------------------------------------------------------------------------

def run_baseline_eval(
    test_pairs: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Run the ticket parsing task against the baseline OpenAI model.
    Returns aggregate scores.
    """
    from openai import OpenAI
    from finetune.scripts.prepare_dataset import SYSTEM_PROMPT

    client = OpenAI(api_key=api_key)
    evaluator = TicketParserEvaluator()

    scores = []
    json_parse_rates = []

    logger.info("Running baseline eval with %s on %d examples...", model, len(test_pairs))

    for pair in test_pairs:
        messages = pair["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        ground_truth_str = next(m["content"] for m in messages if m["role"] == "assistant")
        ground_truth = json.loads(ground_truth_str)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            predicted = response.choices[0].message.content
        except Exception as exc:
            logger.warning("Baseline API call failed: %s", exc)
            predicted = "{}"

        result = evaluator.evaluate_output(predicted, ground_truth)
        scores.append(result["overall_score"])
        json_parse_rates.append(1.0 if result["json_valid"] else 0.0)

    return {
        "model": model,
        "avg_field_accuracy": sum(scores) / len(scores) if scores else 0.0,
        "json_parse_rate": sum(json_parse_rates) / len(json_parse_rates) if json_parse_rates else 0.0,
        "n_samples": len(test_pairs),
    }


# ---------------------------------------------------------------------------
# Fine-tuned model eval (local inference)
# ---------------------------------------------------------------------------

def run_finetuned_eval(
    test_pairs: list[dict],
    model_path: str | Path,
) -> dict:
    """
    Run ticket parsing eval against the fine-tuned model.
    Requires GPU.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    logger.info("Loading fine-tuned model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
    )

    evaluator = TicketParserEvaluator()
    scores = []
    json_parse_rates = []

    logger.info("Running fine-tuned eval on %d examples...", len(test_pairs))

    for pair in test_pairs:
        messages = pair["messages"]
        # Use only system + user messages as input
        input_messages = [m for m in messages if m["role"] != "assistant"]
        ground_truth_str = next(m["content"] for m in messages if m["role"] == "assistant")
        ground_truth = json.loads(ground_truth_str)

        try:
            result_msgs = pipe(input_messages)
            predicted = result_msgs[0]["generated_text"][-1]["content"]
        except Exception as exc:
            logger.warning("Fine-tuned inference failed: %s", exc)
            predicted = "{}"

        result = evaluator.evaluate_output(predicted, ground_truth)
        scores.append(result["overall_score"])
        json_parse_rates.append(1.0 if result["json_valid"] else 0.0)

    return {
        "model": str(model_path),
        "avg_field_accuracy": sum(scores) / len(scores) if scores else 0.0,
        "json_parse_rate": sum(json_parse_rates) / len(json_parse_rates) if json_parse_rates else 0.0,
        "n_samples": len(test_pairs),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_comparison(baseline: dict, finetuned: dict) -> None:
    improvement = finetuned["avg_field_accuracy"] - baseline["avg_field_accuracy"]
    status = "✅ IMPROVED" if improvement > 0 else "❌ REGRESSED"

    print(f"\n{'═'*60}")
    print(f"  Fine-tune Evaluation Report  {status}")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>8}")
    print(f"{'─'*60}")
    print(
        f"  {'Field Accuracy':<25} "
        f"{baseline['avg_field_accuracy']:>11.3f} "
        f"{finetuned['avg_field_accuracy']:>11.3f} "
        f"{improvement:>+8.3f}"
    )
    print(
        f"  {'JSON Parse Rate':<25} "
        f"{baseline['json_parse_rate']:>11.3f} "
        f"{finetuned['json_parse_rate']:>11.3f} "
        f"{finetuned['json_parse_rate'] - baseline['json_parse_rate']:>+8.3f}"
    )
    print(f"{'─'*60}")
    print(f"  Samples evaluated: {finetuned['n_samples']}")
    print(f"{'═'*60}\n")

    if improvement > 0:
        print("✅ Fine-tuned model outperforms baseline — safe to deploy.")
    else:
        print("⚠️  Fine-tuned model does not improve on baseline.")
        print("   Consider more training data or hyperparameter tuning.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Meesho ticket parser")
    parser.add_argument(
        "--model",
        default="finetune/merged_model/qwen25_7b_meesho_merged",
        help="Path to merged fine-tuned model",
    )
    parser.add_argument(
        "--eval-data",
        default="finetune/data/eval_sft_pairs.jsonl",
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare fine-tuned model against gpt-4o-mini baseline",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of eval samples to use (default: 50)",
    )
    args = parser.parse_args()

    # Load eval data
    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        print(f"❌ Eval data not found: {eval_path}")
        print("   Run: python finetune/scripts/prepare_dataset.py")
        exit(1)

    test_pairs = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_pairs.append(json.loads(line))
    test_pairs = test_pairs[:args.n_samples]
    logger.info("Loaded %d eval pairs", len(test_pairs))

    # Run fine-tuned eval
    finetuned_results = run_finetuned_eval(test_pairs, args.model)

    if args.compare:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("❌ OPENAI_API_KEY not set — cannot run baseline comparison")
            exit(1)
        baseline_results = run_baseline_eval(test_pairs, api_key)
        print_comparison(baseline_results, finetuned_results)
    else:
        print(f"\n── Fine-tuned Model Results ──────────────────────────")
        print(f"  Field Accuracy:  {finetuned_results['avg_field_accuracy']:.3f}")
        print(f"  JSON Parse Rate: {finetuned_results['json_parse_rate']:.3f}")
        print(f"  Samples:         {finetuned_results['n_samples']}")
