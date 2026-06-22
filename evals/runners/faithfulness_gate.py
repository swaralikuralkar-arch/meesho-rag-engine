"""
evals/runners/faithfulness_gate.py
=====================================
CI deployment gate based on Ragas Faithfulness score.

Reads the latest eval report and enforces the 85% faithfulness
threshold. Exits with code 1 if the threshold is not met — this
causes GitHub Actions to fail the workflow and block deployment.

Usage (standalone):
    python evals/runners/faithfulness_gate.py
    python evals/runners/faithfulness_gate.py --report evals/reports/latest_eval.json
    python evals/runners/faithfulness_gate.py --threshold 0.90

Usage (from run_evals.py):
    gate = FaithfulnessGate(threshold=0.85)
    gate.evaluate(report)   # raises GateFailure on failure
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD   = 0.85
DEFAULT_REPORT_PATH = "evals/reports/latest_eval.json"


# ---------------------------------------------------------------------------
# Gate failure exception
# ---------------------------------------------------------------------------

class GateFailure(Exception):
    """Raised when the faithfulness gate is not met."""
    pass


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------

@dataclass
class CategoryBreakdown:
    category: str
    avg_faithfulness: float
    num_triplets: int
    passed: bool


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class FaithfulnessGate:
    """
    Evaluates an EvalReport against the faithfulness threshold.

    Parameters
    ----------
    threshold : float
        Minimum required average faithfulness score. Default: 0.85.
    require_all_categories : bool
        If True, every category must individually meet the threshold.
        If False, only the overall average is checked. Default: False.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        require_all_categories: bool = False,
    ) -> None:
        self.threshold = threshold
        self.require_all_categories = require_all_categories

    def evaluate(self, report: dict) -> None:
        """
        Evaluate an EvalReport dict against the gate.

        Raises GateFailure if the threshold is not met.
        Prints a detailed breakdown to stdout.
        """
        avg_faithfulness = report.get("avg_faithfulness", 0.0)
        total            = report.get("total_triplets", 0)
        scored           = report.get("scored_triplets", 0)
        failed           = report.get("failed_triplets", 0)
        decline_rate     = report.get("decline_rate", 0.0)
        timestamp        = report.get("timestamp", "unknown")

        print(f"\n{'═'*60}")
        print(f"  Faithfulness Gate Evaluation")
        print(f"  Timestamp: {timestamp}")
        print(f"{'─'*60}")
        print(f"  Triplets:       {total} total | {scored} scored | {failed} failed")
        print(f"  Decline rate:   {decline_rate:.1%}")
        print(f"{'─'*60}")
        print(f"  Avg Faithfulness:  {avg_faithfulness:.4f}")
        print(f"  Gate Threshold:    {self.threshold:.4f}")
        print(f"  Gate Status:       {'✅ PASS' if avg_faithfulness >= self.threshold else '❌ FAIL'}")

        # Per-category breakdown
        breakdowns = self._compute_category_breakdown(report)
        if breakdowns:
            print(f"{'─'*60}")
            print(f"  Per-Category Breakdown:")
            for b in breakdowns:
                status = "✅" if b.passed else "❌"
                print(
                    f"    {status} {b.category:<20} "
                    f"faithfulness={b.avg_faithfulness:.3f}  "
                    f"n={b.num_triplets}"
                )

        print(f"{'═'*60}\n")

        # Gate logic
        failures: list[str] = []

        if avg_faithfulness < self.threshold:
            failures.append(
                f"Overall faithfulness {avg_faithfulness:.4f} "
                f"is below threshold {self.threshold:.4f}"
            )

        if self.require_all_categories:
            for b in breakdowns:
                if not b.passed:
                    failures.append(
                        f"Category '{b.category}' faithfulness {b.avg_faithfulness:.4f} "
                        f"is below threshold {self.threshold:.4f}"
                    )

        if failures:
            msg = "GATE FAILED:\n" + "\n".join(f"  - {f}" for f in failures)
            raise GateFailure(msg)

        logger.info("Faithfulness gate passed: %.4f >= %.4f", avg_faithfulness, self.threshold)

    def _compute_category_breakdown(self, report: dict) -> list[CategoryBreakdown]:
        """Compute per-category average faithfulness from triplet results."""
        results = report.get("results", [])
        if not results:
            return []

        # Group by category
        by_category: dict[str, list[float]] = {}
        for r in results:
            cat = r.get("category", "unknown")
            faith = r.get("faithfulness")
            if faith is not None:
                by_category.setdefault(cat, []).append(faith)

        breakdowns = []
        for cat, scores in sorted(by_category.items()):
            avg = sum(scores) / len(scores)
            breakdowns.append(CategoryBreakdown(
                category=cat,
                avg_faithfulness=avg,
                num_triplets=len(scores),
                passed=avg >= self.threshold,
            ))
        return breakdowns


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Meesho RAG CI Faithfulness Gate"
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help=f"Path to eval report JSON (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Faithfulness threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--require-all-categories",
        action="store_true",
        help="Require every category to individually meet the threshold",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        print("   Run 'python evals/run_evals.py' first to generate the report.")
        sys.exit(1)

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    gate = FaithfulnessGate(
        threshold=args.threshold,
        require_all_categories=args.require_all_categories,
    )

    try:
        gate.evaluate(report)
        print("✅ Deployment gate passed — safe to deploy.")
        sys.exit(0)
    except GateFailure as exc:
        print(f"\n❌ {exc}")
        print("\n🚫 Deployment blocked. Fix retrieval or prompt issues and re-run evals.")
        sys.exit(1)


if __name__ == "__main__":
    main()
