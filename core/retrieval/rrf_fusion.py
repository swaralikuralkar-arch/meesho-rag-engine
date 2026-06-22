"""
core/retrieval/rrf_fusion.py
==============================
Reciprocal Rank Fusion (RRF) implementation for the Meesho RAG pipeline.

RRF merges two ranked lists (dense vector results + BM25 results) into a
single unified ranking without requiring score normalization.

Formula:
  RRF_score(doc) = Σ 1 / (k + rank_i(doc))

  Where:
    - rank_i(doc) is the 1-based rank of the document in list i
    - k is a smoothing constant (default: 60, from the original RRF paper)
    - The sum is over all ranked lists containing the document

Why RRF over score normalization?
  Dense cosine similarity scores and BM25 scores live in completely
  different ranges and distributions. Normalizing them introduces
  assumptions about their distributions. RRF uses only rank position,
  which is distribution-agnostic and empirically outperforms most
  score-fusion approaches on retrieval benchmarks.

  k=60 is the standard default. Lower k values (e.g. 10) give more
  weight to top-ranked results; higher values (e.g. 100) flatten the
  distribution. 60 is a safe default for this corpus size.

Public API:
  RecipRankFusion.fuse(dense_results, bm25_results, top_k) → list[dict]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RecipRankFusion:
    """
    Reciprocal Rank Fusion over two result lists.

    Parameters
    ----------
    k : int
        RRF smoothing constant. Default: 60.
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        dense_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Merge dense and BM25 result lists using RRF.

        Parameters
        ----------
        dense_results : list[dict]
            Ordered results from QdrantVectorStore.search().
            Schema: [{"chunk_id", "content", "score", "metadata"}, ...]
        bm25_results : list[dict]
            Ordered results from BM25Index.search().
            Same schema as dense_results.
        top_k : int
            Number of fused results to return. Default: 25.

        Returns
        -------
        list[dict]
            RRF-ranked results. Each dict adds "rrf_score",
            "dense_rank", and "bm25_rank" fields for observability.
        """
        # Map chunk_id → full result dict (dense takes precedence for content)
        chunk_registry: dict[str, dict[str, Any]] = {}

        for result in dense_results:
            cid = result["chunk_id"]
            if cid not in chunk_registry:
                chunk_registry[cid] = {**result, "dense_rank": None, "bm25_rank": None}

        for result in bm25_results:
            cid = result["chunk_id"]
            if cid not in chunk_registry:
                chunk_registry[cid] = {**result, "dense_rank": None, "bm25_rank": None}

        # Assign ranks (1-based)
        for rank, result in enumerate(dense_results, start=1):
            chunk_registry[result["chunk_id"]]["dense_rank"] = rank

        for rank, result in enumerate(bm25_results, start=1):
            chunk_registry[result["chunk_id"]]["bm25_rank"] = rank

        # Compute RRF score for each chunk
        rrf_scores: dict[str, float] = {}
        for cid, data in chunk_registry.items():
            score = 0.0
            if data["dense_rank"] is not None:
                score += 1.0 / (self.k + data["dense_rank"])
            if data["bm25_rank"] is not None:
                score += 1.0 / (self.k + data["bm25_rank"])
            rrf_scores[cid] = score

        # Sort by RRF score descending
        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        ranked_ids = ranked_ids[:top_k]

        results: list[dict[str, Any]] = []
        for cid in ranked_ids:
            entry = chunk_registry[cid].copy()
            entry["rrf_score"] = rrf_scores[cid]
            # Replace raw score with rrf_score for downstream consumers
            entry["score"] = rrf_scores[cid]
            results.append(entry)

        logger.debug(
            "RRF fusion: %d dense + %d bm25 → %d candidates (top_k=%d)",
            len(dense_results), len(bm25_results), len(results), top_k,
        )
        return results
