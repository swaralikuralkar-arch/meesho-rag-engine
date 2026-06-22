"""
core/retrieval/reranker.py
============================
Cross-encoder reranker for the Meesho RAG pipeline.

Takes the top-25 RRF candidates and re-scores them using a cross-encoder
model that jointly attends to both the query and the candidate chunk.
This is more accurate than bi-encoder retrieval but too slow to run over
the full corpus — hence the two-stage architecture.

Model options (in order of preference):
  1. cross-encoder/ms-marco-MiniLM-L-6-v2 — fast, 22M params, strong
     on factual retrieval tasks. Default.
  2. BAAI/bge-reranker-large — stronger, 560M params, slower.
     Set via RERANKER_MODEL env var or config.
  3. Cohere rerank API — hosted, requires COHERE_API_KEY. Used when
     local compute is constrained.

Output:
  Top-5 chunks re-ranked by cross-encoder score, with the raw
  cross-encoder score stored as "rerank_score" for observability.

Public API:
  CrossEncoderReranker.rerank(query, candidates, top_k=5) → list[dict]
  CohereReranker.rerank(query, candidates, top_k=5) → list[dict]
  get_reranker(config) → CrossEncoderReranker | CohereReranker
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reranker protocol (shared interface)
# ---------------------------------------------------------------------------

class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# Local cross-encoder reranker
# ---------------------------------------------------------------------------

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence-transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace model name. Default: ms-marco-MiniLM-L-6-v2.
    device : str
        Torch device. Default: "cpu".
    max_length : int
        Max token length for query+passage. Default: 512.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s on device=%s", model_name, device)
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )
        logger.info("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Re-score candidates using the cross-encoder and return top_k.

        Parameters
        ----------
        query : str
            Raw user query.
        candidates : list[dict]
            RRF fusion output. Each dict must have "content".
        top_k : int
            Number of results to return after reranking. Default: 5.

        Returns
        -------
        list[dict]
            Top-k candidates with "rerank_score" added and sorted
            descending by rerank_score.
        """
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))

        # Build (query, passage) pairs
        pairs = [(query, c["content"]) for c in candidates]

        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            "CrossEncoder rerank: %d candidates → top %d  "
            "(top score=%.4f, bottom score=%.4f)",
            len(candidates), top_k,
            scored[0]["rerank_score"],
            scored[top_k - 1]["rerank_score"],
        )

        return scored[:top_k]


# ---------------------------------------------------------------------------
# Cohere reranker (hosted API)
# ---------------------------------------------------------------------------

COHERE_RERANK_MODEL = "rerank-english-v3.0"


class CohereReranker:
    """
    Hosted Cohere reranker via the Cohere API.

    Requires: pip install cohere
    Requires: COHERE_API_KEY environment variable.

    Use this when local compute is unavailable or when latency SLAs
    are relaxed enough to tolerate an API round-trip (~200-400ms).
    """

    def __init__(
        self,
        model: str = COHERE_RERANK_MODEL,
        api_key: str | None = None,
    ) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package not installed. Run: pip install cohere"
            )

        _api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not _api_key:
            raise ValueError(
                "COHERE_API_KEY not set. Export it or pass api_key= explicitly."
            )

        self._client = cohere.Client(_api_key)
        self._model = model
        logger.info("CohereReranker initialized with model: %s", model)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))
        documents = [c["content"] for c in candidates]

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        results: list[dict[str, Any]] = []
        for hit in response.results:
            candidate = candidates[hit.index].copy()
            candidate["rerank_score"] = float(hit.relevance_score)
            results.append(candidate)

        logger.debug(
            "CohereReranker: %d candidates → top %d", len(candidates), top_k
        )
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_reranker(
    backend: str = "local",
    model_name: str | None = None,
    device: str = "cpu",
    cohere_api_key: str | None = None,
) -> CrossEncoderReranker | CohereReranker:
    """
    Instantiate the correct reranker based on config.

    Parameters
    ----------
    backend : str
        "local" → CrossEncoderReranker
        "cohere" → CohereReranker
    model_name : str | None
        Override the default model name.
    device : str
        Torch device for local reranker.
    cohere_api_key : str | None
        Cohere API key (for backend="cohere").
    """
    if backend == "cohere":
        return CohereReranker(
            model=model_name or COHERE_RERANK_MODEL,
            api_key=cohere_api_key,
        )
    else:
        return CrossEncoderReranker(
            model_name=model_name or DEFAULT_CROSS_ENCODER_MODEL,
            device=device,
        )
