"""
core/observability/langsmith_tracer.py
========================================
LangSmith tracing integration for the Meesho RAG engine.

Traces every query end-to-end using LangSmith's RunTree API —
no LangChain required. Each query produces a parent run with
three child spans:

  meesho_rag_query  (parent)
  ├── retrieval     (dense + BM25 + RRF + rerank)
  ├── generation    (prompt build + LLM call)
  └── validation    (citation enforcement + schema validation)

Each span captures:
  - inputs / outputs
  - latency (ms)
  - token counts
  - rerank scores
  - error details if any step fails

Env vars required (loaded from .env):
  LANGCHAIN_API_KEY       — LangSmith API key
  LANGCHAIN_TRACING_V2    — must be "true"
  LANGCHAIN_PROJECT       — project name (e.g. "meesho-rag-engine")

Public API:
  LangSmithTracer.trace_query(fn, query, **kwargs) → RAGResponse
  LangSmithTracer.trace_retrieval(fn, query, **kwargs) → list[RetrievedChunk]
  @LangSmithTracer.traced_span(name) → decorator
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith client setup
# ---------------------------------------------------------------------------

def _init_langsmith():
    try:
        from langsmith import Client
        from langsmith.run_trees import RunTree

        api_key = os.environ.get("LANGCHAIN_API_KEY", "")
        if not api_key:
            logger.warning(
                "LANGCHAIN_API_KEY not set — LangSmith tracing disabled."
            )
            return None, None

        client = Client(api_key=api_key)
        logger.info("LangSmith client initialized. Project: %s",
                    os.environ.get("LANGCHAIN_PROJECT", "default"))
        return client, RunTree
    except ImportError:
        logger.warning("langsmith not installed — tracing disabled.")
        return None, None

_LS_CLIENT, _RunTree = _init_langsmith()
TRACING_ENABLED = _LS_CLIENT is not None


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class LangSmithTracer:
    """
    LangSmith tracer for the Meesho RAG engine.

    Parameters
    ----------
    project_name : str
        LangSmith project name. Defaults to LANGCHAIN_PROJECT env var.
    enabled : bool
        Set False to disable tracing (e.g. in unit tests).
    """

    def __init__(
        self,
        project_name: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.project_name = (
            project_name
            or os.environ.get("LANGCHAIN_PROJECT", "meesho-rag-engine")
        )
        self.enabled = enabled and TRACING_ENABLED

        if not self.enabled:
            logger.info("LangSmithTracer: tracing disabled.")

    # ------------------------------------------------------------------
    # Full query trace (parent span)
    # ------------------------------------------------------------------

    def trace_query(
        self,
        query: str,
        retrieval_fn: Callable,
        generation_fn: Callable,
        validation_fn: Callable,
        retrieval_kwargs: dict | None = None,
        generation_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """
        Trace a complete RAG query with three child spans.

        Parameters
        ----------
        query : str
            User query string.
        retrieval_fn : Callable
            Function that takes query → list[RetrievedChunk].
        generation_fn : Callable
            Function that takes (query, chunks) → RAGResponse.
        validation_fn : Callable
            Function that takes (response, chunks) → RAGResponse.

        Returns
        -------
        dict with keys: response, chunks, trace_id, latency_ms
        """
        t_total_start = time.perf_counter()

        if not self.enabled:
            chunks = retrieval_fn(query, **(retrieval_kwargs or {}))
            response = generation_fn(query, chunks, **(generation_kwargs or {}))
            validated = validation_fn(response, chunks)
            return {
                "response": validated,
                "chunks": chunks,
                "trace_id": None,
                "latency_ms": (time.perf_counter() - t_total_start) * 1000,
            }

        # Create parent run
        parent_run = _RunTree(
            name="meesho_rag_query",
            run_type="chain",
            project_name=self.project_name,
            inputs={"query": query},
        )
        parent_run.post()

        chunks = None
        response = None
        validated = None

        try:
            # ── Span 1: Retrieval ──────────────────────────────────────
            chunks = self._run_child_span(
                parent_run=parent_run,
                name="retrieval",
                run_type="retriever",
                inputs={"query": query},
                fn=retrieval_fn,
                fn_args=(query,),
                fn_kwargs=retrieval_kwargs or {},
                output_serializer=lambda c: {
                    "num_chunks": len(c),
                    "chunk_ids": [ch.chunk_id for ch in c],
                    "rerank_scores": [round(ch.rerank_score, 4) for ch in c],
                    "content_types": [ch.content_type for ch in c],
                    "latency_ms": round(c[0].retrieval_latency_ms, 1) if c else 0,
                },
            )

            # ── Span 2: Generation ─────────────────────────────────────
            response = self._run_child_span(
                parent_run=parent_run,
                name="generation",
                run_type="llm",
                inputs={
                    "query": query,
                    "num_context_chunks": len(chunks) if chunks else 0,
                },
                fn=generation_fn,
                fn_args=(query, chunks),
                fn_kwargs=generation_kwargs or {},
                output_serializer=lambda r: {
                    "answer": r.answer[:200],
                    "confidence": r.confidence,
                    "is_decline": r.is_decline,
                    "metric_value": r.metric_value,
                    "num_citations": len(r.citations),
                },
            )

            # ── Span 3: Validation ─────────────────────────────────────
            validated = self._run_child_span(
                parent_run=parent_run,
                name="citation_validation",
                run_type="chain",
                inputs={"num_citations": len(response.citations) if response else 0},
                fn=validation_fn,
                fn_args=(response, chunks),
                fn_kwargs={},
                output_serializer=lambda r: {
                    "is_decline": r.is_decline,
                    "final_citations": len(r.citations),
                    "confidence": r.confidence,
                },
            )

            total_latency_ms = (time.perf_counter() - t_total_start) * 1000

            parent_run.end(outputs={
                "answer": validated.answer[:200] if validated else "",
                "is_decline": validated.is_decline if validated else True,
                "confidence": validated.confidence if validated else 0.0,
                "total_latency_ms": round(total_latency_ms, 1),
            })
            parent_run.patch()

            logger.info(
                "LangSmith trace posted. run_id=%s  latency=%.1fms",
                parent_run.id, total_latency_ms
            )

            return {
                "response": validated,
                "chunks": chunks,
                "trace_id": str(parent_run.id),
                "latency_ms": total_latency_ms,
            }

        except Exception as exc:
            total_latency_ms = (time.perf_counter() - t_total_start) * 1000
            parent_run.end(error=str(exc))
            parent_run.patch()
            logger.error("RAG query trace failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Child span helper
    # ------------------------------------------------------------------

    def _run_child_span(
        self,
        parent_run: Any,
        name: str,
        run_type: str,
        inputs: dict,
        fn: Callable,
        fn_args: tuple,
        fn_kwargs: dict,
        output_serializer: Callable,
    ) -> Any:
        """Execute a function inside a LangSmith child span."""
        child_run = parent_run.create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
        )
        child_run.post()

        t_start = time.perf_counter()
        try:
            result = fn(*fn_args, **fn_kwargs)
            latency_ms = (time.perf_counter() - t_start) * 1000

            outputs = output_serializer(result)
            outputs["latency_ms"] = round(latency_ms, 1)

            child_run.end(outputs=outputs)
            child_run.patch()
            return result

        except Exception as exc:
            child_run.end(error=str(exc))
            child_run.patch()
            raise

    # ------------------------------------------------------------------
    # Decorator for standalone span tracing
    # ------------------------------------------------------------------

    def traced_span(self, name: str, run_type: str = "chain"):
        """
        Decorator that wraps a function in a standalone LangSmith run.

        Usage:
            tracer = LangSmithTracer()

            @tracer.traced_span("my_function")
            def my_function(x, y):
                return x + y
        """
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return fn(*args, **kwargs)

                run = _RunTree(
                    name=name,
                    run_type=run_type,
                    project_name=self.project_name,
                    inputs={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                )
                run.post()
                t_start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    run.end(outputs={
                        "result": str(result)[:200],
                        "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    })
                    run.patch()
                    return result
                except Exception as exc:
                    run.end(error=str(exc))
                    run.patch()
                    raise
            return wrapper
        return decorator
