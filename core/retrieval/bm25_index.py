"""
core/retrieval/bm25_index.py
==============================
BM25 sparse keyword index for the Meesho RAG hybrid search pipeline.

Why BM25 alongside dense vectors?
  Dense embeddings excel at semantic similarity but underperform on
  exact keyword matches — e.g. "RTO penalty ₹50" or "commission tier 3"
  where the user knows the exact term. BM25 fills this gap: it scores
  documents by term frequency × inverse document frequency, rewarding
  exact keyword matches regardless of semantic proximity.

  For Meesho's supplier docs, this matters most for:
    - Specific policy codes (e.g. "Section 4.2", "Tier-3 supplier")
    - Exact monetary values (e.g. "₹85 per kg", "18% commission")
    - Product category names that may not cluster semantically

Implementation:
  - Uses rank-bm25 (BM25Okapi variant) with k1=1.5, b=0.75.
  - Tokenization: lowercase, punctuation-stripped word tokens.
    Numbers are kept intact (critical for rate/slab queries).
  - Index is persisted to disk as a pickle file so it survives restarts
    without re-indexing the full corpus.
  - chunk_ids are stored in parallel with the BM25 corpus so search
    results map back to the same chunk_id namespace as the vector store.

Public API:
  BM25Index.build(chunk_dicts)         → build from scratch
  BM25Index.save(path)                 → persist to disk
  BM25Index.load(path)                 → restore from disk
  BM25Index.search(query, top_k) → list[dict]  (same schema as VectorStore.search)
"""

from __future__ import annotations

import logging
import pickle
import re
import string
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Punctuation to strip (keep ₹ and % — they are meaningful in Meesho docs)
_STRIP_PUNCT = string.punctuation.replace("₹", "").replace("%", "")
_PUNCT_TABLE = str.maketrans("", "", _STRIP_PUNCT)


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    - Lowercase
    - Strip most punctuation (keep ₹ and %)
    - Split on whitespace
    - Remove empty tokens
    """
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t]
    return tokens


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Persistent BM25 index over ChunkRecord content.

    Parameters
    ----------
    k1 : float
        BM25 term frequency saturation parameter. Default: 1.5.
    b : float
        BM25 document length normalization parameter. Default: 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_contents: list[str] = []
        self._chunk_metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunk_dicts: list[dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of ChunkRecord.to_dict() outputs.

        Parameters
        ----------
        chunk_dicts : list[dict]
            Each dict must have "chunk_id", "content", and "metadata".
        """
        if not chunk_dicts:
            raise ValueError("Cannot build BM25 index from empty chunk list.")

        logger.info("Building BM25 index over %d chunks...", len(chunk_dicts))

        self._chunk_ids = [c["chunk_id"] for c in chunk_dicts]
        self._chunk_contents = [c["content"] for c in chunk_dicts]
        self._chunk_metadata = [c.get("metadata", {}) for c in chunk_dicts]

        tokenized_corpus = [tokenize(text) for text in self._chunk_contents]

        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        logger.info(
            "BM25 index built. Corpus size: %d docs, avg tokens/doc: %.1f",
            len(tokenized_corpus),
            sum(len(t) for t in tokenized_corpus) / max(len(tokenized_corpus), 1),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the index to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "k1": self.k1,
            "b": self.b,
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
            "chunk_contents": self._chunk_contents,
            "chunk_metadata": self._chunk_metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("BM25 index saved to: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        """Restore a previously saved BM25 index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(k1=state["k1"], b=state["b"])
        instance._bm25 = state["bm25"]
        instance._chunk_ids = state["chunk_ids"]
        instance._chunk_contents = state["chunk_contents"]
        instance._chunk_metadata = state["chunk_metadata"]

        logger.info(
            "BM25 index loaded from %s (%d docs)", path, len(instance._chunk_ids)
        )
        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 25,
        filter_doc_id: str | None = None,
        filter_content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Score and rank chunks against a query using BM25.

        Parameters
        ----------
        query : str
            Raw query string. Tokenized internally.
        top_k : int
            Number of results to return. Default: 25.
        filter_doc_id : str | None
            Optional: restrict results to a specific doc_id.
        filter_content_type : str | None
            Optional: restrict results to a content_type.

        Returns
        -------
        list[dict]
            Same schema as QdrantVectorStore.search() for easy merging:
            [{"chunk_id": str, "content": str, "score": float, "metadata": dict}]
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning("BM25 query tokenized to empty list: %r", query)
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Build (index, score) pairs and apply filters
        candidates: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self._chunk_metadata[idx]
            if filter_doc_id and meta.get("doc_id") != filter_doc_id:
                continue
            if filter_content_type and meta.get("content_type") != filter_content_type:
                continue
            candidates.append((idx, float(score)))

        # Sort by score descending, take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        return [
            {
                "chunk_id": self._chunk_ids[idx],
                "content": self._chunk_contents[idx],
                "score": score,
                "metadata": self._chunk_metadata[idx],
            }
            for idx, score in candidates
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def is_built(self) -> bool:
        return self._bm25 is not None
