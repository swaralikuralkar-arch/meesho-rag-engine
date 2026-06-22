"""
core/observability/metrics_exporter.py
=========================================
In-process metrics collection and reporting for the Meesho RAG engine.

Tracks per-query telemetry and computes aggregate statistics:
  - P50 / P95 tail latencies (total, retrieval, generation)
  - Cost per token (using OpenAI pricing for gpt-4o-mini)
  - Decline rate
  - Average confidence score
  - Cache hit rate (future extension point)

Design:
  - All metrics are stored in-memory in a circular buffer (last 1000
    queries) to keep memory bounded.
  - Stats are computed on-demand (not streaming) — suitable for a
    dashboard endpoint that polls every 30s.
  - Metrics are also written to a JSONL log file for offline analysis
    and the CI eval gate.

Public API:
  MetricsExporter.record(query_result)
  MetricsExporter.get_stats() → MetricsSnapshot
  MetricsExporter.export_jsonl(path)
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI pricing constants (gpt-4o-mini, as of 2025)
# Update these if switching models.
# ---------------------------------------------------------------------------
COST_PER_INPUT_TOKEN  = 0.00000015   # $0.15 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 0.00000060   # $0.60 per 1M output tokens

MAX_BUFFER_SIZE = 1000  # circular buffer size


# ---------------------------------------------------------------------------
# Per-query record
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    """Telemetry captured for a single RAG query."""
    timestamp: float
    query: str
    trace_id: str | None

    # Latencies (ms)
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float

    # Token counts
    input_tokens: int
    output_tokens: int

    # Response quality
    confidence: float
    is_decline: bool
    num_citations: int
    num_chunks_retrieved: int

    # Rerank diagnostics
    top_rerank_score: float
    top_content_type: str

    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens  * COST_PER_INPUT_TOKEN +
            self.output_tokens * COST_PER_OUTPUT_TOKEN
        )


# ---------------------------------------------------------------------------
# Aggregate snapshot
# ---------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    """Aggregate statistics over the last N queries."""
    sample_size: int
    window_start: float
    window_end: float

    # Latency percentiles (ms)
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    p50_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    p50_generation_latency_ms: float
    p95_generation_latency_ms: float

    # Cost
    total_cost_usd: float
    avg_cost_per_query_usd: float
    avg_input_tokens: float
    avg_output_tokens: float

    # Quality
    decline_rate: float          # 0.0 - 1.0
    avg_confidence: float
    avg_citations_per_query: float
    avg_chunks_retrieved: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pretty_print(self) -> str:
        return (
            f"\n{'═'*55}\n"
            f"  Meesho RAG Metrics Snapshot  (n={self.sample_size})\n"
            f"{'─'*55}\n"
            f"  Latency (total)      P50={self.p50_total_latency_ms:.0f}ms  "
            f"P95={self.p95_total_latency_ms:.0f}ms\n"
            f"  Latency (retrieval)  P50={self.p50_retrieval_latency_ms:.0f}ms  "
            f"P95={self.p95_retrieval_latency_ms:.0f}ms\n"
            f"  Latency (generation) P50={self.p50_generation_latency_ms:.0f}ms  "
            f"P95={self.p95_generation_latency_ms:.0f}ms\n"
            f"{'─'*55}\n"
            f"  Cost/query           ${self.avg_cost_per_query_usd:.6f}\n"
            f"  Total cost           ${self.total_cost_usd:.4f}\n"
            f"  Avg input tokens     {self.avg_input_tokens:.0f}\n"
            f"  Avg output tokens    {self.avg_output_tokens:.0f}\n"
            f"{'─'*55}\n"
            f"  Decline rate         {self.decline_rate:.1%}\n"
            f"  Avg confidence       {self.avg_confidence:.3f}\n"
            f"  Avg citations        {self.avg_citations_per_query:.1f}\n"
            f"  Avg chunks retrieved {self.avg_chunks_retrieved:.1f}\n"
            f"{'═'*55}\n"
        )


# ---------------------------------------------------------------------------
# Metrics exporter
# ---------------------------------------------------------------------------

class MetricsExporter:
    """
    Collects and aggregates telemetry for the Meesho RAG engine.

    Parameters
    ----------
    log_path : str | Path | None
        If set, each query record is appended to this JSONL file.
    buffer_size : int
        Maximum number of recent queries to keep in memory.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        buffer_size: int = MAX_BUFFER_SIZE,
    ) -> None:
        self._buffer: deque[QueryMetrics] = deque(maxlen=buffer_size)
        self._log_path = Path(log_path) if log_path else None

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(
        self,
        query: str,
        trace_result: dict[str, Any],
        input_tokens: int = 0,
        output_tokens: int = 0,
        generation_latency_ms: float = 0.0,
    ) -> QueryMetrics:
        """
        Record telemetry for a completed query.

        Parameters
        ----------
        query : str
            Original user query.
        trace_result : dict
            Output from LangSmithTracer.trace_query() containing:
            response, chunks, trace_id, latency_ms.
        input_tokens : int
            Prompt token count from LLM response.usage.
        output_tokens : int
            Completion token count from LLM response.usage.
        generation_latency_ms : float
            Time spent in the LLM call specifically.
        """
        response = trace_result.get("response")
        chunks   = trace_result.get("chunks") or []
        trace_id = trace_result.get("trace_id")
        total_latency_ms = trace_result.get("latency_ms", 0.0)

        retrieval_latency_ms = (
            chunks[0].retrieval_latency_ms if chunks else 0.0
        )

        metrics = QueryMetrics(
            timestamp=time.time(),
            query=query[:100],
            trace_id=trace_id,
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            confidence=response.confidence if response else 0.0,
            is_decline=response.is_decline if response else True,
            num_citations=len(response.citations) if response else 0,
            num_chunks_retrieved=len(chunks),
            top_rerank_score=chunks[0].rerank_score if chunks else 0.0,
            top_content_type=chunks[0].content_type if chunks else "unknown",
        )

        self._buffer.append(metrics)

        if self._log_path:
            self._append_to_log(metrics)

        logger.debug(
            "Metrics recorded: latency=%.1fms  confidence=%.2f  decline=%s",
            total_latency_ms, metrics.confidence, metrics.is_decline
        )
        return metrics

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, last_n: int | None = None) -> MetricsSnapshot | None:
        """
        Compute aggregate statistics over recent queries.

        Parameters
        ----------
        last_n : int | None
            If set, compute over the last N queries only.

        Returns
        -------
        MetricsSnapshot or None if no data recorded yet.
        """
        records = list(self._buffer)
        if last_n:
            records = records[-last_n:]

        if not records:
            logger.warning("No metrics recorded yet.")
            return None

        def _percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            return statistics.quantiles(data, n=100)[int(p) - 1]

        total_latencies     = [r.total_latency_ms for r in records]
        retrieval_latencies = [r.retrieval_latency_ms for r in records]
        generation_latencies= [r.generation_latency_ms for r in records]
        costs               = [r.estimated_cost_usd() for r in records]

        return MetricsSnapshot(
            sample_size=len(records),
            window_start=records[0].timestamp,
            window_end=records[-1].timestamp,

            p50_total_latency_ms=_percentile(total_latencies, 50),
            p95_total_latency_ms=_percentile(total_latencies, 95),
            p50_retrieval_latency_ms=_percentile(retrieval_latencies, 50),
            p95_retrieval_latency_ms=_percentile(retrieval_latencies, 95),
            p50_generation_latency_ms=_percentile(generation_latencies, 50),
            p95_generation_latency_ms=_percentile(generation_latencies, 95),

            total_cost_usd=sum(costs),
            avg_cost_per_query_usd=statistics.mean(costs),
            avg_input_tokens=statistics.mean([r.input_tokens for r in records]),
            avg_output_tokens=statistics.mean([r.output_tokens for r in records]),

            decline_rate=sum(1 for r in records if r.is_decline) / len(records),
            avg_confidence=statistics.mean([r.confidence for r in records]),
            avg_citations_per_query=statistics.mean([r.num_citations for r in records]),
            avg_chunks_retrieved=statistics.mean([r.num_chunks_retrieved for r in records]),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(self, path: str | Path) -> None:
        """Export all buffered metrics to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(asdict(record)) + "\n")
        logger.info("Exported %d metrics records to %s", len(self._buffer), path)

    def _append_to_log(self, metrics: QueryMetrics) -> None:
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
