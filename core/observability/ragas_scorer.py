"""
core/observability/ragas_scorer.py
=====================================
Real-time Ragas evaluation scoring for the Meesho RAG engine.

Computes two core Ragas metrics per query:
  - Faithfulness:     Does the answer stay grounded in the context?
                      Score 0.0-1.0. Penalizes hallucinations.
  - Context Recall:   Does the retrieved context cover the ground truth?
                      Score 0.0-1.0. Penalizes retrieval gaps.

These scores feed into:
  - LangSmith traces (attached as feedback scores)
  - MetricsExporter (tracked over time)
  - Phase 3 CI gate (85% faithfulness threshold blocks deployment)

Note on async: Ragas uses async LLM calls internally. This module
wraps them in a synchronous interface for compatibility with the
main pipeline. For high-throughput use, call score_async() directly.

Public API:
  RagasScorer.score(query, answer, contexts, ground_truth) -> RagasResult
  RagasScorer.score_from_rag_response(query, response, chunks) -> RagasResult
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RagasResult:
    """Ragas evaluation scores for a single query."""
    faithfulness: float | None       # 0.0-1.0, None if scoring failed
    context_recall: float | None     # 0.0-1.0, None if no ground truth
    context_precision: float | None  # 0.0-1.0, bonus metric
    error: str | None = None         # Error message if scoring failed

    def passed_ci_gate(self, threshold: float = 0.85) -> bool:
        """Returns True if faithfulness meets the CI deployment threshold."""
        if self.faithfulness is None:
            return False
        return self.faithfulness >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "error": self.error,
            "passed_ci_gate": self.passed_ci_gate(),
        }


# ---------------------------------------------------------------------------
# Ragas scorer
# ---------------------------------------------------------------------------

class RagasScorer:
    """
    Wraps Ragas evaluation metrics for use in the Meesho RAG pipeline.

    Parameters
    ----------
    openai_api_key : str | None
        OpenAI API key for Ragas LLM calls. Falls back to OPENAI_API_KEY.
    model : str
        LLM model used by Ragas for faithfulness scoring. Default: gpt-4o-mini.
    enabled : bool
        Set False to skip scoring (e.g. in unit tests or when no API key).
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
        enabled: bool = True,
    ) -> None:
        self._api_key = (
            openai_api_key
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._model = model
        self.enabled = enabled and bool(self._api_key)

        if not self.enabled:
            logger.info("RagasScorer: disabled (no API key or enabled=False).")
            return

        self._init_ragas()

    def _init_ragas(self) -> None:
        """Initialize Ragas metrics and LLM wrapper."""
        try:
            from ragas.metrics import faithfulness, context_recall, context_precision
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self._model,
                openai_api_key=self._api_key,
                temperature=0,
            )
            llm_wrapper = LangchainLLMWrapper(llm)

            self._faithfulness = faithfulness
            self._context_recall = context_recall
            self._context_precision = context_precision

            # Inject LLM into metrics
            self._faithfulness.llm = llm_wrapper
            self._context_recall.llm = llm_wrapper
            self._context_precision.llm = llm_wrapper

            logger.info("RagasScorer initialized with model: %s", self._model)

        except Exception as exc:
            logger.warning("RagasScorer init failed: %s — scoring disabled.", exc)
            self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RagasResult:
        """
        Score a single query-answer-context triple.

        Parameters
        ----------
        query : str
            User query.
        answer : str
            Generated answer from the RAG engine.
        contexts : list[str]
            Retrieved chunk contents used to generate the answer.
        ground_truth : str | None
            Reference answer for context_recall scoring.
            If None, context_recall is skipped.

        Returns
        -------
        RagasResult
        """
        if not self.enabled:
            return RagasResult(
                faithfulness=None,
                context_recall=None,
                context_precision=None,
                error="Scorer disabled",
            )

        try:
            return asyncio.run(
                self._score_async(query, answer, contexts, ground_truth)
            )
        except RuntimeError:
            # Already inside an event loop (e.g. Jupyter) — use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self._score_async(query, answer, contexts, ground_truth)
                )
            except Exception as exc:
                logger.error("Ragas async scoring failed: %s", exc)
                return RagasResult(
                    faithfulness=None,
                    context_recall=None,
                    context_precision=None,
                    error=str(exc),
                )
        except Exception as exc:
            logger.error("Ragas scoring failed: %s", exc)
            return RagasResult(
                faithfulness=None,
                context_recall=None,
                context_precision=None,
                error=str(exc),
            )

    def score_from_rag_response(
        self,
        query: str,
        response: Any,           # RAGResponse
        chunks: list[Any],       # list[RetrievedChunk]
        ground_truth: str | None = None,
    ) -> RagasResult:
        """
        Convenience wrapper — extracts contexts from RAGResponse + chunks.

        Parameters
        ----------
        query : str
            User query.
        response : RAGResponse
            Validated RAGResponse from the engine.
        chunks : list[RetrievedChunk]
            Retrieved chunks used in generation.
        ground_truth : str | None
            Optional reference answer for context_recall.
        """
        if response.is_decline:
            logger.debug("Skipping Ragas scoring for decline response.")
            return RagasResult(
                faithfulness=None,
                context_recall=None,
                context_precision=None,
                error="Skipped: decline response",
            )

        contexts = [
            chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
            for chunk in chunks
        ]

        return self.score(
            query=query,
            answer=response.answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # Async scorer
    # ------------------------------------------------------------------

    async def _score_async(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> RagasResult:
        """Async implementation — called by score() via asyncio.run()."""
        from datasets import Dataset

        # Build Ragas dataset
        data: dict[str, list] = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        faithfulness_score = None
        context_recall_score = None
        context_precision_score = None

        # Faithfulness (always computed)
        try:
            f_result = await self._faithfulness.ascore(
                row={
                    "question": query,
                    "answer": answer,
                    "contexts": contexts,
                }
            )
            faithfulness_score = float(f_result)
        except Exception as exc:
            logger.warning("Faithfulness scoring failed: %s", exc)

        # Context Recall (only if ground truth provided)
        if ground_truth:
            try:
                cr_result = await self._context_recall.ascore(
                    row={
                        "question": query,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": ground_truth,
                    }
                )
                context_recall_score = float(cr_result)
            except Exception as exc:
                logger.warning("Context recall scoring failed: %s", exc)

        # Context Precision
        try:
            cp_result = await self._context_precision.ascore(
                row={
                    "question": query,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth or answer,
                }
            )
            context_precision_score = float(cp_result)
        except Exception as exc:
            logger.warning("Context precision scoring failed: %s", exc)

        result = RagasResult(
            faithfulness=faithfulness_score,
            context_recall=context_recall_score,
            context_precision=context_precision_score,
        )

        logger.info(
            "Ragas scores — faithfulness=%.3f  context_recall=%s  "
            "context_precision=%s",
            faithfulness_score or 0,
            f"{context_recall_score:.3f}" if context_recall_score else "N/A",
            f"{context_precision_score:.3f}" if context_precision_score else "N/A",
        )

        return result
