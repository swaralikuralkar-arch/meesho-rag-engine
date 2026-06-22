"""
core/retrieval/vector_store.py
================================
Qdrant vector store wrapper for Meesho RAG engine.

Responsibilities:
  - Initialize and manage the Qdrant collection with named dense vectors.
  - Embed chunks using BGE-large-en-v1.5 (1024-dim, cosine distance).
  - Upsert ChunkRecords as Qdrant points with full metadata as payload.
  - Execute dense-only vector search (used by hybrid_search.py).

BGE embedding notes:
  - Passages (at index time) use the prefix:
      "Represent this sentence for searching relevant passages: "
  - Queries (at search time) use the prefix:
      "Represent this question for retrieving relevant documents: "
  - normalize_embeddings=True is required for cosine similarity to work
    correctly with dot-product distance in Qdrant.

Collection schema:
  - One named vector: "dense" (size=1024, distance=Cosine)
  - Payload fields mirror ChunkRecord.metadata exactly so Qdrant's
    payload filtering can be used for doc_id / content_type scoping.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSE_VECTOR_NAME = "dense"
DENSE_VECTOR_SIZE = 1024          # bge-large-en-v1.5 output dimension
COLLECTION_NAME   = "meesho_rag_v1"

BGE_MODEL_NAME    = "BAAI/bge-large-en-v1.5"
BGE_PASSAGE_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_QUERY_PREFIX   = "Represent this question for retrieving relevant documents: "


# ---------------------------------------------------------------------------
# Embedder (singleton pattern — model load is expensive)
# ---------------------------------------------------------------------------

class BGEEmbedder:
    """
    Wraps SentenceTransformer BGE model with correct prefix injection.
    Loaded once per process; shared across VectorStore instances.
    """

    _instance: BGEEmbedder | None = None

    def __new__(cls, model_name: str = BGE_MODEL_NAME, device: str = "cpu"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(model_name, device)
        return cls._instance

    def _init(self, model_name: str, device: str) -> None:
        logger.info("Loading BGE model: %s on device=%s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._device = device
        logger.info("BGE model loaded.")

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        prefixed = [BGE_PASSAGE_PREFIX + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = BGE_QUERY_PREFIX + query
        vec = self._model.encode(
            prefixed,
            normalize_embeddings=True,
        )
        return vec.tolist()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class QdrantVectorStore:
    """
    Manages the Meesho RAG Qdrant collection.

    Parameters
    ----------
    host : str
        Qdrant server host. Default: "localhost".
    port : int
        Qdrant REST port. Default: 6333.
    grpc_port : int
        Qdrant gRPC port. Default: 6334.
    prefer_grpc : bool
        Use gRPC for upsert/search (lower latency). Default: True.
    collection_name : str
        Qdrant collection name. Default: COLLECTION_NAME.
    embedding_device : str
        Torch device for BGE model. Default: "cpu".
    in_memory : bool
        If True, uses an in-memory Qdrant instance (for testing/CI).
        Ignores host/port when True.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        collection_name: str = COLLECTION_NAME,
        embedding_device: str = "cpu",
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self._embedder = BGEEmbedder(device=embedding_device)

        if in_memory:
            logger.info("Initializing in-memory Qdrant client (test mode)")
            self._client = QdrantClient(":memory:")
        else:
            logger.info("Connecting to Qdrant at %s:%d", host, port)
            self._client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
            )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with the dense named vector config.
        If recreate=True, drops and recreates the collection.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if self.collection_name in existing:
            if recreate:
                logger.warning("Recreating collection: %s", self.collection_name)
                self._client.delete_collection(self.collection_name)
            else:
                logger.info("Collection %r already exists — skipping create.", self.collection_name)
                return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: qmodels.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                )
            },
        )
        logger.info("Created collection: %s", self.collection_name)

    def collection_exists(self) -> bool:
        existing = [c.name for c in self._client.get_collections().collections]
        return self.collection_name in existing

    def count(self) -> int:
        return self._client.count(self.collection_name).count

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunk_dicts: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        """
        Embed and upsert a list of ChunkRecord.to_dict() outputs into Qdrant.

        Parameters
        ----------
        chunk_dicts : list[dict]
            Each dict must have "content" (str) and "metadata" (dict).
            chunk_id is used as the Qdrant point ID (converted to UUID).
        batch_size : int
            Embedding batch size. Default: 32.
        """
        if not chunk_dicts:
            logger.warning("upsert_chunks called with empty list — skipping.")
            return

        logger.info("Upserting %d chunks into collection %r", len(chunk_dicts), self.collection_name)

        # Process in batches
        for i in range(0, len(chunk_dicts), batch_size):
            batch = chunk_dicts[i:i + batch_size]
            texts = [c["content"] for c in batch]

            vectors = self._embedder.embed_passages(texts, batch_size=batch_size)

            points = []
            for chunk_dict, vector in zip(batch, vectors):
                point_id = self._chunk_id_to_uuid(chunk_dict["chunk_id"])
                payload = {
                    "chunk_id": chunk_dict["chunk_id"],
                    "content": chunk_dict["content"],
                    **chunk_dict["metadata"],
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector={DENSE_VECTOR_NAME: vector},
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.debug("  Upserted batch %d-%d", i, i + len(batch))

        logger.info("Upsert complete. Collection size: %d", self.count())

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 25,
        filter_doc_id: str | None = None,
        filter_content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense vector search against the Qdrant collection.

        Parameters
        ----------
        query : str
            Raw query string. BGE query prefix is applied internally.
        top_k : int
            Number of results to return. Default: 25 (pre-rerank pool).
        filter_doc_id : str | None
            Optional: restrict search to a specific document.
        filter_content_type : str | None
            Optional: restrict to "table", "prose", "list_item", etc.

        Returns
        -------
        list[dict]
            Each dict has keys: chunk_id, content, score, metadata.
        """
        query_vector = self._embedder.embed_query(query)

        # Build optional payload filter
        qdrant_filter = self._build_filter(filter_doc_id, filter_content_type)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=(DENSE_VECTOR_NAME, query_vector),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id", str(hit.id)),
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "metadata": {
                    k: v for k, v in hit.payload.items()
                    if k not in ("chunk_id", "content")
                },
            }
            for hit in results
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        """
        Convert a chunk_id string like "doc::0::1" to a deterministic UUID.
        Using uuid5 ensures the same chunk_id always maps to the same UUID,
        making upserts idempotent.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    @staticmethod
    def _build_filter(
        filter_doc_id: str | None,
        filter_content_type: str | None,
    ) -> qmodels.Filter | None:
        conditions = []
        if filter_doc_id:
            conditions.append(
                qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=filter_doc_id),
                )
            )
        if filter_content_type:
            conditions.append(
                qmodels.FieldCondition(
                    key="content_type",
                    match=qmodels.MatchValue(value=filter_content_type),
                )
            )
        if not conditions:
            return None
        return qmodels.Filter(must=conditions)
