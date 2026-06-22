"""
evals/run_evals.py
====================
Single entry point for the Meesho RAG evaluation suite.

Usage:
    # Full eval run
    python evals/run_evals.py

    # Fast run (first 3 triplets only — for PR checks)
    python evals/run_evals.py --fast

    # Custom threshold
    python evals/run_evals.py --threshold 0.90

    # Skip gate (generate report only)
    python evals/run_evals.py --no-gate

Exit codes:
    0 — all evals passed, gate met
    1 — evals ran but gate failed (blocks deployment)
    2 — eval run itself failed (infrastructure error)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evals")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meesho RAG Eval Suite")
    parser.add_argument(
        "--dataset",
        default="evals/golden_dataset/supplier_triplets.jsonl",
        help="Path to golden dataset JSONL",
    )
    parser.add_argument(
        "--output",
        default="evals/reports/latest_eval.json",
        help="Output path for eval report JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Faithfulness gate threshold (default: 0.85)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only first 3 triplets (for fast PR checks)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip the faithfulness gate check (report only)",
    )
    parser.add_argument(
        "--require-all-categories",
        action="store_true",
        help="Require every category to meet the threshold individually",
    )
    args = parser.parse_args()

    # ── Step 1: Run evals ────────────────────────────────────────────
    try:
        from evals.runners.ragas_eval_runner import RagasEvalRunner

        runner = RagasEvalRunner(
            faithfulness_threshold=args.threshold,
            max_triplets=3 if args.fast else None,
        )
        report = runner.run(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    except Exception as exc:
        logger.error("Eval run failed with infrastructure error: %s", exc)
        sys.exit(2)

    # ── Step 2: Enforce gate ─────────────────────────────────────────
    if args.no_gate:
        logger.info("Gate check skipped (--no-gate).")
        sys.exit(0)

    try:
        from evals.runners.faithfulness_gate import FaithfulnessGate, GateFailure

        gate = FaithfulnessGate(
            threshold=args.threshold,
            require_all_categories=args.require_all_categories,
        )
        gate.evaluate(report.to_dict())
        logger.info("✅ Eval gate passed.")
        sys.exit(0)

    except GateFailure as exc:
        logger.error("❌ Eval gate failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
