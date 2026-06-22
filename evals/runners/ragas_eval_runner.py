"""
evals/runners/ragas_eval_runner.py
=====================================
Evaluation runner for the Meesho RAG engine CI gate.

Loads the golden dataset (JSONL triplets), runs each query through
the RAG pipeline in offline mode (using stored context rather than
live retrieval), and scores each response using Ragas.

Offline eval mode:
  Rather than running live retrieval (which requires Qdrant + BGE),
  the eval runner uses the stored context from each golden triplet
  directly. This makes evals fast, deterministic, and runnable in CI
  without any infrastructure dependencies.

  The context from each triplet is wrapped into a mock RetrievedChunk
  so it passes through the generation and citation enforcement layers
  unchanged.

Output:
  EvalReport dataclass with per-query scores and aggregate statistics.
  Written to evals/reports/ as JSON for historical tracking.

Public API:
  RagasEvalRunner.run(dataset_path, output_path) -> EvalReport
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock retrieved chunk (wraps golden context for offline eval)
# ---------------------------------------------------------------------------

@dataclass
class MockRetrievedChunk:
    """Wraps a golden context string as a RetrievedChunk for eval use."""
    chunk_id: str
    content: str
    rerank_score: float = 1.0
    rrf_score: float = 1.0
    dense_rank: int = 1
    bm25_rank: int = 1
    doc_id: str = "golden_dataset"
    doc_type: str = "eval"
    page_number: int = 1
    section_header: str | None = None
    table_index: int | None = None
    content_type: str = "prose"
    token_count: int = 0
    retrieval_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class TripletResult:
    """Evaluation result for a single golden triplet."""
    query: str
    ground_truth: str
    generated_answer: str
    is_decline: bool
    confidence: float
    faithfulness: float | None
    context_recall: float | None
    context_precision: float | None
    doc_id: str
    category: str
    difficulty: str
    latency_ms: float
    error: str | None = None

    def passed(self, faithfulness_threshold: float = 0.85) -> bool:
        if self.faithfulness is None:
            return False
        return self.faithfulness >= faithfulness_threshold


@dataclass
class EvalReport:
    """Aggregate evaluation report over the full golden dataset."""
    timestamp: str
    dataset_path: str
    total_triplets: int
    scored_triplets: int
    failed_triplets: int

    avg_faithfulness: float
    avg_context_recall: float
    avg_context_precision: float
    decline_rate: float
    avg_confidence: float
    avg_latency_ms: float

    faithfulness_gate_threshold: float
    passed_gate: bool

    results: list[TripletResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def pretty_print(self) -> str:
        status = "✅ PASSED" if self.passed_gate else "❌ FAILED"
        return (
            f"\n{'═'*60}\n"
            f"  Meesho RAG Eval Report  {status}\n"
            f"{'─'*60}\n"
            f"  Dataset:          {self.dataset_path}\n"
            f"  Triplets:         {self.total_triplets} total, "
            f"{self.scored_triplets} scored, {self.failed_triplets} failed\n"
            f"{'─'*60}\n"
            f"  Faithfulness:     {self.avg_faithfulness:.3f}  "
            f"(gate: {self.faithfulness_gate_threshold:.2f})\n"
            f"  Context Recall:   {self.avg_context_recall:.3f}\n"
            f"  Context Precision:{self.avg_context_precision:.3f}\n"
            f"{'─'*60}\n"
            f"  Decline Rate:     {self.decline_rate:.1%}\n"
            f"  Avg Confidence:   {self.avg_confidence:.3f}\n"
            f"  Avg Latency:      {self.avg_latency_ms:.0f}ms\n"
            f"{'═'*60}\n"
        )


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

class RagasEvalRunner:
    """
    Runs the Meesho RAG eval suite against the golden dataset.

    Parameters
    ----------
    openai_api_key : str | None
        OpenAI API key for LLM calls (generation + Ragas scoring).
    model : str
        LLM model for generation and Ragas. Default: gpt-4o-mini.
    faithfulness_threshold : float
        CI gate threshold. Default: 0.85.
    max_triplets : int | None
        If set, only evaluate the first N triplets (for fast CI runs).
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
        faithfulness_threshold: float = 0.85,
        max_triplets: int | None = None,
    ) -> None:
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._threshold = faithfulness_threshold
        self._max_triplets = max_triplets

        # Lazy imports — keeps startup fast
        self._prompt_builder = None
        self._output_parser = None
        self._citation_enforcer = None
        self._ragas_scorer = None

    def _init_pipeline(self) -> None:
        """Initialize generation pipeline components."""
        from core.generation.prompt_builder import PromptBuilder
        from core.generation.structured_output import StructuredOutputParser
        from core.generation.citation_enforcer import CitationEnforcer
        from core.observability.ragas_scorer import RagasScorer

        self._prompt_builder = PromptBuilder()
        self._output_parser = StructuredOutputParser(
            model=self._model,
            api_key=self._api_key,
        )
        self._citation_enforcer = CitationEnforcer(strict=False)
        self._ragas_scorer = RagasScorer(
            openai_api_key=self._api_key,
            model=self._model,
        )
        logger.info("Eval pipeline initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_path: str | Path = "evals/golden_dataset/supplier_triplets.jsonl",
        output_path: str | Path | None = "evals/reports/latest_eval.json",
    ) -> EvalReport:
        """
        Run the full eval suite and return an EvalReport.

        Parameters
        ----------
        dataset_path : str | Path
            Path to the golden dataset JSONL file.
        output_path : str | Path | None
            If set, write the EvalReport JSON here.
        """
        self._init_pipeline()

        triplets = self._load_dataset(dataset_path)
        if self._max_triplets:
            triplets = triplets[:self._max_triplets]

        logger.info("Running eval on %d triplets...", len(triplets))

        results: list[TripletResult] = []
        for i, triplet in enumerate(triplets, start=1):
            logger.info("  [%d/%d] %s", i, len(triplets), triplet["query"][:60])
            result = self._eval_triplet(triplet)
            results.append(result)

        report = self._build_report(dataset_path, results)

        if output_path:
            self._save_report(report, output_path)

        print(report.pretty_print())
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _eval_triplet(self, triplet: dict) -> TripletResult:
        """Run a single golden triplet through the pipeline."""
        t_start = time.perf_counter()

        query        = triplet["query"]
        context      = triplet["context"]
        ground_truth = triplet["ground_truth"]
        doc_id       = triplet.get("doc_id", "unknown")
        category     = triplet.get("category", "unknown")
        difficulty   = triplet.get("difficulty", "unknown")

        # Wrap context as a mock chunk
        chunk = MockRetrievedChunk(
            chunk_id=f"{doc_id}::eval::0",
            content=context,
            doc_id=doc_id,
            section_header=triplet.get("section"),
        )

        try:
            # Build prompt
            prompt_pair = self._prompt_builder.build(query, [chunk])

            # Generate
            response = self._output_parser.parse(query, prompt_pair, [chunk])

            # Validate citations
            validated = self._citation_enforcer.validate(response, [chunk])

            # Score with Ragas
            ragas_result = self._ragas_scorer.score_from_rag_response(
                query=query,
                response=validated,
                chunks=[chunk],
                ground_truth=ground_truth,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000

            return TripletResult(
                query=query,
                ground_truth=ground_truth,
                generated_answer=validated.answer,
                is_decline=validated.is_decline,
                confidence=validated.confidence,
                faithfulness=ragas_result.faithfulness,
                context_recall=ragas_result.context_recall,
                context_precision=ragas_result.context_precision,
                doc_id=doc_id,
                category=category,
                difficulty=difficulty,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.error("Triplet eval failed: %s", exc)
            return TripletResult(
                query=query,
                ground_truth=ground_truth,
                generated_answer="",
                is_decline=True,
                confidence=0.0,
                faithfulness=None,
                context_recall=None,
                context_precision=None,
                doc_id=doc_id,
                category=category,
                difficulty=difficulty,
                latency_ms=latency_ms,
                error=str(exc),
            )

    @staticmethod
    def _load_dataset(path: str | Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {path}")
        triplets = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    triplets.append(json.loads(line))
        logger.info("Loaded %d triplets from %s", len(triplets), path)
        return triplets

    def _build_report(
        self,
        dataset_path: str | Path,
        results: list[TripletResult],
    ) -> EvalReport:
        from datetime import datetime, timezone

        scored = [r for r in results if r.faithfulness is not None]
        failed = [r for r in results if r.error is not None]

        def safe_avg(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else 0.0

        avg_faithfulness = safe_avg([r.faithfulness for r in scored])

        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_path=str(dataset_path),
            total_triplets=len(results),
            scored_triplets=len(scored),
            failed_triplets=len(failed),
            avg_faithfulness=avg_faithfulness,
            avg_context_recall=safe_avg([r.context_recall for r in scored]),
            avg_context_precision=safe_avg([r.context_precision for r in scored]),
            decline_rate=sum(1 for r in results if r.is_decline) / len(results),
            avg_confidence=safe_avg([r.confidence for r in results]),
            avg_latency_ms=safe_avg([r.latency_ms for r in results]),
            faithfulness_gate_threshold=self._threshold,
            passed_gate=avg_faithfulness >= self._threshold,
            results=results,
        )

    @staticmethod
    def _save_report(report: EvalReport, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Eval report saved to: %s", path)
