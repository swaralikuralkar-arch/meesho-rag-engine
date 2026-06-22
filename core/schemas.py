"""
core/schemas.py
================
Pydantic v2 schemas for the Meesho RAG engine output contract.

This file defines the single source of truth for what the RAG engine
returns. Every layer — generation, evals, API, CI gate — imports from
here. Never define output shapes inline in other modules.

Output contract design principles:
  1. Deterministic citations: every answer must reference at least one
     source chunk by chunk_id. The LLM cannot fabricate citations.
  2. Graceful decline: if retrieved context does not contain enough
     information to answer confidently, the engine returns a structured
     DECLINE response rather than hallucinating.
  3. Metric extraction: for quantitative queries (commission rates,
     weight slabs, payment timelines), the numeric value is extracted
     into a dedicated field so downstream systems can parse it without
     regex hacks.
  4. Confidence scoring: a self-reported 0.0-1.0 confidence score
     from the LLM, used by the eval harness to threshold responses.

Schema hierarchy:
  RAGResponse
  ├── answer: str
  ├── confidence: float (0.0 - 1.0)
  ├── is_decline: bool
  ├── decline_reason: str | None
  ├── metric_value: str | None       (e.g. "18%", "₹85/kg", "7 days")
  ├── metric_unit: str | None        (e.g. "%", "₹/kg", "days")
  └── citations: list[Citation]
        ├── chunk_id: str
        ├── doc_id: str
        ├── section_header: str | None
        ├── page_number: int
        ├── content_type: str
        └── exact_quote_context: str  (verbatim excerpt from chunk)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    TABLE     = "table"
    PROSE     = "prose"
    LIST_ITEM = "list_item"
    HEADING   = "heading"


class DeclineReason(str, Enum):
    NO_CONTEXT         = "no_relevant_context_found"
    LOW_CONFIDENCE     = "confidence_below_threshold"
    CONFLICTING_CONTEXT = "conflicting_information_in_context"
    OUT_OF_SCOPE       = "query_out_of_scope"


# ---------------------------------------------------------------------------
# Citation schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """
    A single source attribution linking an answer claim to a retrieved chunk.

    The exact_quote_context field must be a verbatim substring of the
    chunk content — the citation_enforcer.py validates this constraint.
    """

    chunk_id: str = Field(
        description="Unique chunk identifier in format doc_id::block_idx::chunk_idx"
    )
    doc_id: str = Field(
        description="Source document identifier"
    )
    section_header: str | None = Field(
        default=None,
        description="Section heading under which this chunk appears"
    )
    page_number: int = Field(
        default=1,
        description="Page number in the source document (1 for HTML sources)",
        ge=1,
    )
    content_type: str = Field(
        default="prose",
        description="Block type: table | prose | list_item | heading"
    )
    exact_quote_context: str = Field(
        description=(
            "A short verbatim excerpt (1-3 sentences or 1-2 table rows) from "
            "the source chunk that directly supports the answer. Must be an "
            "exact substring of the chunk content, not paraphrased."
        ),
        min_length=10,
    )

    @field_validator("exact_quote_context")
    @classmethod
    def quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("exact_quote_context cannot be empty or whitespace")
        return v.strip()


# ---------------------------------------------------------------------------
# Main RAG response schema
# ---------------------------------------------------------------------------

class RAGResponse(BaseModel):
    """
    The complete structured output of the Meesho RAG engine for a single query.

    Returned by core/generation/structured_output.py and validated by
    core/generation/citation_enforcer.py before being sent to the caller.
    """

    answer: str = Field(
        description=(
            "The natural language answer to the user query. "
            "If is_decline=True, this contains a user-friendly explanation "
            "of why the query cannot be answered."
        ),
        min_length=1,
    )

    confidence: float = Field(
        description=(
            "Self-reported confidence score (0.0-1.0). "
            "Scores below 0.5 should trigger a decline in production."
        ),
        ge=0.0,
        le=1.0,
    )

    is_decline: bool = Field(
        default=False,
        description="True if the engine is declining to answer due to insufficient context.",
    )

    decline_reason: DeclineReason | None = Field(
        default=None,
        description="Structured reason code for the decline. Required if is_decline=True.",
    )

    metric_value: str | None = Field(
        default=None,
        description=(
            "For quantitative queries, the extracted numeric or categorical value. "
            "Examples: '18%', '₹85', '7 days', 'Tier-3'. None for non-metric queries."
        ),
    )

    metric_unit: str | None = Field(
        default=None,
        description=(
            "Unit of the metric_value if applicable. "
            "Examples: '%', '₹/kg', 'days', 'INR'. None if not applicable."
        ),
    )

    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "List of source chunks that support the answer. "
            "Must be non-empty unless is_decline=True."
        ),
    )

    query: str = Field(
        description="The original user query that generated this response.",
        min_length=1,
    )

    # ------------------------------------------------------------------
    # Cross-field validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_decline_consistency(self) -> RAGResponse:
        """
        If is_decline=True, decline_reason must be set.
        If is_decline=False, citations must be non-empty.
        """
        if self.is_decline:
            if self.decline_reason is None:
                raise ValueError(
                    "decline_reason must be set when is_decline=True"
                )
        else:
            if not self.citations:
                raise ValueError(
                    "citations must be non-empty when is_decline=False. "
                    "If the context is insufficient, set is_decline=True."
                )
        return self

    @model_validator(mode="after")
    def validate_confidence_decline_alignment(self) -> RAGResponse:
        """
        A non-decline response with confidence < 0.3 is suspicious —
        force it to a decline to prevent low-quality answers reaching users.
        """
        if not self.is_decline and self.confidence < 0.3:
            self.is_decline = True
            self.decline_reason = DeclineReason.LOW_CONFIDENCE
            self.citations = []
        return self

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def make_decline(
        cls,
        query: str,
        reason: DeclineReason = DeclineReason.NO_CONTEXT,
        message: str | None = None,
    ) -> RAGResponse:
        """
        Factory method for a well-formed decline response.
        Use this instead of constructing DeclineReason manually.
        """
        answer = message or (
            "I was unable to find sufficient information in Meesho's supplier "
            "documentation to answer this question confidently. Please refer to "
            "the official Meesho Supplier Panel or contact Meesho support."
        )
        return cls(
            query=query,
            answer=answer,
            confidence=0.0,
            is_decline=True,
            decline_reason=reason,
            citations=[],
        )


# ---------------------------------------------------------------------------
# LLM raw output schema (intermediate — before citation validation)
# ---------------------------------------------------------------------------

class LLMRawOutput(BaseModel):
    """
    The raw structured output parsed directly from the LLM response,
    before citation_enforcer validates quote accuracy.

    This is an intermediate type used internally by structured_output.py.
    The caller always receives a validated RAGResponse, never this type.
    """

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_decline: bool = False
    decline_reason: str | None = None
    metric_value: str | None = None
    metric_unit: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
