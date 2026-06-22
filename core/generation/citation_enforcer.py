"""
core/generation/citation_enforcer.py
=======================================
Post-generation citation validation layer.

The LLM is instructed to include exact_quote_context as a verbatim
excerpt from the source chunk. This module verifies that instruction
was followed — catching hallucinated or paraphrased quotes before
the response reaches the caller.

Validation checks per citation:
  1. chunk_id exists in the retrieved chunks pool
  2. exact_quote_context is a substring of the source chunk content
     (case-insensitive, whitespace-normalized)
  3. exact_quote_context meets minimum length (10 chars)

On failure:
  - STRICT mode  → raises CitationError (use in CI eval gate)
  - LENIENT mode → removes invalid citations, logs warnings,
                   downgrades confidence score, may trigger decline
                   if no valid citations remain (use in production)

Public API:
  CitationEnforcer.validate(response, retrieved_chunks) → RAGResponse
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.schemas import RAGResponse, Citation, DeclineReason

logger = logging.getLogger(__name__)

MIN_QUOTE_LENGTH  = 10
CONFIDENCE_PENALTY = 0.2   # applied per invalid citation in lenient mode


class CitationError(Exception):
    """Raised in strict mode when a citation fails validation."""
    pass


class CitationEnforcer:
    """
    Validates citations in a RAGResponse against the retrieved chunks.

    Parameters
    ----------
    strict : bool
        If True, raises CitationError on any invalid citation.
        If False (default), removes invalid citations and adjusts confidence.
    min_quote_length : int
        Minimum character length for exact_quote_context. Default: 10.
    """

    def __init__(
        self,
        strict: bool = False,
        min_quote_length: int = MIN_QUOTE_LENGTH,
    ) -> None:
        self.strict = strict
        self.min_quote_length = min_quote_length

    def validate(
        self,
        response: RAGResponse,
        retrieved_chunks: list[Any],
    ) -> RAGResponse:
        """
        Validate all citations in a RAGResponse.

        Parameters
        ----------
        response : RAGResponse
            The response to validate (output of StructuredOutputParser).
        retrieved_chunks : list[RetrievedChunk | dict]
            The chunks that were passed to the LLM.

        Returns
        -------
        RAGResponse
            Validated response. In lenient mode, invalid citations are
            removed. In strict mode, CitationError is raised.
        """
        if response.is_decline:
            # No citations to validate on a decline response
            return response

        # Build chunk content lookup: chunk_id → content string
        content_lookup = self._build_content_lookup(retrieved_chunks)

        valid_citations: list[Citation] = []
        invalid_count = 0

        for citation in response.citations:
            error = self._validate_citation(citation, content_lookup)

            if error is None:
                valid_citations.append(citation)
            else:
                invalid_count += 1
                msg = (
                    f"Citation validation failed for chunk_id={citation.chunk_id!r}: "
                    f"{error}"
                )
                if self.strict:
                    raise CitationError(msg)
                else:
                    logger.warning(msg)

        if invalid_count == 0:
            logger.debug("All %d citations validated successfully.", len(valid_citations))
            return response

        # Lenient mode: reconstruct response with valid citations only
        new_confidence = max(
            0.0,
            response.confidence - (invalid_count * CONFIDENCE_PENALTY)
        )

        if not valid_citations:
            # All citations were invalid — force a decline
            logger.warning(
                "All %d citations failed validation — converting to decline.",
                invalid_count
            )
            return RAGResponse.make_decline(
                query=response.query,
                reason=DeclineReason.NO_CONTEXT,
                message=(
                    "The system was unable to verify the sources for this response. "
                    "Please consult the official Meesho Supplier Panel directly."
                ),
            )

        # Return updated response with valid citations and reduced confidence
        data = response.model_dump()
        data["citations"] = [c.model_dump() for c in valid_citations]
        data["confidence"] = new_confidence
        return RAGResponse.model_validate(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_citation(
        self,
        citation: Citation,
        content_lookup: dict[str, str],
    ) -> str | None:
        """
        Validate a single citation.
        Returns None if valid, or an error message string if invalid.
        """
        # Check 1: chunk_id must exist in retrieved chunks
        if citation.chunk_id not in content_lookup:
            return f"chunk_id not found in retrieved chunks pool"

        # Check 2: quote must meet minimum length
        quote = citation.exact_quote_context.strip()
        if len(quote) < self.min_quote_length:
            return (
                f"exact_quote_context too short: "
                f"{len(quote)} chars (min {self.min_quote_length})"
            )

        # Check 3: quote must be a substring of the source chunk
        chunk_content = content_lookup[citation.chunk_id]
        if not self._is_substring(quote, chunk_content):
            return (
                f"exact_quote_context is not a verbatim substring of the source chunk. "
                f"Quote: {quote[:80]!r}"
            )

        return None

    @staticmethod
    def _is_substring(quote: str, content: str) -> bool:
        """
        Check if quote is a substring of content.
        Uses whitespace normalization and case-insensitivity to handle
        minor formatting differences (extra spaces, newlines).
        """
        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", text).strip().lower()

        return normalize(quote) in normalize(content)

    @staticmethod
    def _build_content_lookup(retrieved_chunks: list[Any]) -> dict[str, str]:
        """
        Build a chunk_id → content string map from retrieved chunks.
        Supports both RetrievedChunk dataclass and dict formats.
        """
        lookup: dict[str, str] = {}
        for chunk in retrieved_chunks:
            if hasattr(chunk, "chunk_id"):
                lookup[chunk.chunk_id] = chunk.content
            elif isinstance(chunk, dict):
                cid = chunk.get("chunk_id", "")
                content = chunk.get("content", "")
                if cid:
                    lookup[cid] = content
        return lookup
