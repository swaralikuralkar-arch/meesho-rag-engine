"""
core/retrieval/hybrid_search.py
=================================
Hybrid search orchestrator for the Meesho RAG pipeline.

This is the single entry point called by the RAG engine at query time.
It coordinates the full retrieval pipeline:

  1. Dense search    → QdrantVectorStore.search(query, top_k=25)
  2. BM25 search     → BM25Index.search(query, top_k=25)
  3. RRF fusion      → RecipRankFusion.fuse(dense, bm25, top_k=25)
  4. Cross-encoder   → Reranker.rerank(query, top_25, top_k=5)

Steps 1 and 2 run concurrently via ThreadPoolExecutor since they are
I/O-bound (Qdrant network call) and CPU-bound (BM25 scoring) respectively
and do not share state.

Output contract:
  list[RetrievedChunk] — ordered by rerank_score descending, top-5.
  Each RetrievedChunk carries all metadata needed for citation
  attribution in the generation layer.

Public API:
  HybridSearchPipeline(vector_store, bm25_index, reranker)
  HybridSearchPipeline.search(query, top_k_retrieval, top_k_final) → list[RetrievedChunk]
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from core.retrieval.vector_store import QdrantVectorStore
from core.retrieval.bm25_index import BM25Index
from core.retrieval.rrf_fusion import RecipRankFusion
from core.retrieval.reranker import CrossEncoderReranker, CohereReranker

logger = logging.getLogger(__name__)

# Type alias for reranker union
AnyReranker = CrossEncoderReranker | CohereReranker


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A single retrieved and reranked chunk, ready for the generation layer.
    Carries all fields needed for deterministic citation attribution.
    """
    chunk_id: str
    content: str
    rerank_score: float
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None

    # Provenance metadata
    doc_id: str
    doc_type: str
    page_number: int
    section_header: str | None
    table_index: int | None
    content_type: str
    token_count: int

    # Retrieval diagnostics
    retrieval_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "rerank_score": self.rerank_score,
            "rrf_score": self.rrf_score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "page_number": self.page_number,
            "section_header": self.section_header,
            "table_index": self.table_index,
            "content_type": self.content_type,
            "token_count": self.token_count,
            "retrieval_latency_ms": self.retrieval_latency_ms,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HybridSearchPipeline:
    """
    Full hybrid retrieval pipeline: dense + BM25 → RRF → rerank.

    Parameters
    ----------
    vector_store : QdrantVectorStore
        Initialized and populated Qdrant vector store.
    bm25_index : BM25Index
        Built BM25 index over the same corpus.
    reranker : AnyReranker
        CrossEncoderReranker or CohereReranker instance.
    rrf_k : int
        RRF smoothing constant. Default: 60.
    use_parallel : bool
        Run dense and BM25 search concurrently. Default: True.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        reranker: AnyReranker,
        rrf_k: int = 60,
        use_parallel: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._rrf = RecipRankFusion(k=rrf_k)
        self._use_parallel = use_parallel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k_retrieval: int = 25,
        top_k_final: int = 5,
        filter_doc_id: str | None = None,
        filter_content_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Execute the full hybrid retrieval pipeline for a query.

        Parameters
        ----------
        query : str
            Raw user query string.
        top_k_retrieval : int
            Number of candidates each retriever returns. Default: 25.
        top_k_final : int
            Number of chunks returned after reranking. Default: 5.
        filter_doc_id : str | None
            Optional payload filter — restrict to a specific document.
        filter_content_type : str | None
            Optional payload filter — restrict to a content type.

        Returns
        -------
        list[RetrievedChunk]
            Top-k chunks ordered by rerank_score descending.
        """
        t_start = time.perf_counter()

        # Step 1 + 2: Dense and BM25 search
        dense_results, bm25_results = self._run_retrieval(
            query, top_k_retrieval, filter_doc_id, filter_content_type
        )

        logger.debug(
            "Retrieval: %d dense + %d bm25 results",
            len(dense_results), len(bm25_results),
        )

        # Step 3: RRF fusion
        fused = self._rrf.fuse(dense_results, bm25_results, top_k=top_k_retrieval)

        # Step 4: Rerank
        reranked = self._reranker.rerank(query, fused, top_k=top_k_final)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        logger.info(
            "HybridSearch: query=%r  results=%d  latency=%.1fms",
            query[:60], len(reranked), latency_ms,
        )

        return self._to_retrieved_chunks(reranked, latency_ms)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_retrieval(
        self,
        query: str,
        top_k: int,
        filter_doc_id: str | None,
        filter_content_type: str | None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Run dense and BM25 search — concurrently if use_parallel=True.
        """
        if self._use_parallel:
            dense_results, bm25_results = None, None

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_dense = executor.submit(
                    self._vector_store.search,
                    query, top_k, filter_doc_id, filter_content_type,
                )
                future_bm25 = executor.submit(
                    self._bm25_index.search,
                    query, top_k, filter_doc_id, filter_content_type,
                )
                dense_results = future_dense.result()
                bm25_results = future_bm25.result()

            return dense_results, bm25_results
        else:
            dense_results = self._vector_store.search(
                query, top_k, filter_doc_id, filter_content_type
            )
            bm25_results = self._bm25_index.search(
                query, top_k, filter_doc_id, filter_content_type
            )
            return dense_results, bm25_results

    @staticmethod
    def _to_retrieved_chunks(
        reranked: list[dict[str, Any]],
        latency_ms: float,
    ) -> list[RetrievedChunk]:
        """Convert reranked dicts to typed RetrievedChunk objects."""
        results = []
        for item in reranked:
            meta = item.get("metadata", {})
            results.append(RetrievedChunk(
                chunk_id=item.get("chunk_id", ""),
                content=item.get("content", ""),
                rerank_score=item.get("rerank_score", 0.0),
                rrf_score=item.get("rrf_score", item.get("score", 0.0)),
                dense_rank=item.get("dense_rank"),
                bm25_rank=item.get("bm25_rank"),
                doc_id=meta.get("doc_id", ""),
                doc_type=meta.get("doc_type", ""),
                page_number=meta.get("page_number", 0),
                section_header=meta.get("section_header"),
                table_index=meta.get("table_index"),
                content_type=meta.get("content_type", ""),
                token_count=meta.get("token_count", 0),
                retrieval_latency_ms=latency_ms,
            ))
        return results
