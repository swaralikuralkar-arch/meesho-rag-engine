"""
core/generation/prompt_builder.py
====================================
Constructs the system prompt and user prompt fed to the LLM.

Design decisions:
  - System prompt is loaded from config/prompts.yaml so it can be
    tuned without code changes (critical for the CI regression gate).
  - Context is formatted differently by content_type:
      table chunks    → wrapped in a code fence for structure preservation
      prose chunks    → plain paragraph with source label
      list_item chunks → plain text with source label
  - Each chunk is labelled with [SOURCE: doc_id | section | page] so
    the LLM can construct exact_quote_context citations accurately.
  - A strict instruction block tells the LLM to decline rather than
    hallucinate if the context is insufficient.
  - Token budget enforcement: if the total context exceeds
    MAX_CONTEXT_TOKENS, chunks are truncated from the bottom
    (lowest rerank_score first) to fit within the budget.

Public API:
  PromptBuilder.build(query, chunks) → PromptPair(system, user)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 6000   # safe ceiling for most LLM context windows
APPROX_CHARS_PER_TOKEN = 4  # approximation for budget enforcement


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class PromptPair:
    system: str
    user: str

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI-compatible messages list."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": self.user},
        ]


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Meesho Supplier Intelligence Assistant — a precise, \
citation-driven AI that answers questions about Meesho's supplier policies, \
commission structures, return/RTO rules, cataloging guidelines, and payment \
settlement processes.

## YOUR CORE RULES

1. **Answer only from the provided context.** Do not use any prior knowledge \
about Meesho. If the context does not contain enough information, you MUST \
decline to answer.

2. **Always cite your sources.** Every factual claim must reference at least \
one source chunk using its exact chunk_id. The `exact_quote_context` field \
must be a verbatim excerpt from that chunk — do not paraphrase it.

3. **Extract metric values.** For questions about rates, fees, timelines, or \
quantities, extract the numeric value into the `metric_value` field and its \
unit into `metric_unit`.

4. **Decline gracefully.** If the context is missing, ambiguous, or \
conflicting, set `is_decline: true` and explain why clearly in the `answer` \
field. Never guess or extrapolate.

5. **Respond ONLY in valid JSON** matching the schema below. No preamble, \
no explanation outside the JSON object.

## OUTPUT SCHEMA

```json
{
  "answer": "<natural language answer, or decline explanation>",
  "confidence": <float 0.0-1.0>,
  "is_decline": <true|false>,
  "decline_reason": "<no_relevant_context_found|confidence_below_threshold|conflicting_information_in_context|query_out_of_scope|null>",
  "metric_value": "<extracted value or null>",
  "metric_unit": "<unit or null>",
  "citations": [
    {
      "chunk_id": "<exact chunk_id from context>",
      "doc_id": "<doc_id from context>",
      "section_header": "<section header or null>",
      "page_number": <int>,
      "content_type": "<table|prose|list_item|heading>",
      "exact_quote_context": "<verbatim excerpt from the chunk>"
    }
  ]
}
```

## DECLINE EXAMPLES

If asked about something not in the context:
```json
{
  "answer": "I could not find information about supplier suspension policies in the provided documentation.",
  "confidence": 0.0,
  "is_decline": true,
  "decline_reason": "no_relevant_context_found",
  "metric_value": null,
  "metric_unit": null,
  "citations": []
}
```
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds system + user prompts from a query and retrieved chunks.

    Parameters
    ----------
    max_context_tokens : int
        Token budget for the context block. Chunks are dropped from
        the bottom (lowest score) if the budget is exceeded.
    system_prompt : str | None
        Override the default system prompt. If None, uses _SYSTEM_PROMPT.
    """

    def __init__(
        self,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        system_prompt: str | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self._system_prompt = system_prompt or _SYSTEM_PROMPT

    def build(
        self,
        query: str,
        chunks: list[Any],   # list[RetrievedChunk] — using Any to avoid circular import
    ) -> PromptPair:
        """
        Build the prompt pair from a query and retrieved chunks.

        Parameters
        ----------
        query : str
            Raw user query.
        chunks : list[RetrievedChunk]
            Ordered list of retrieved chunks (highest rerank_score first).

        Returns
        -------
        PromptPair
            system and user prompt strings.
        """
        if not chunks:
            logger.warning("PromptBuilder called with no chunks — LLM will likely decline.")

        context_block = self._build_context_block(chunks)
        user_prompt = self._build_user_prompt(query, context_block)

        return PromptPair(
            system=self._system_prompt,
            user=user_prompt,
        )

    # ------------------------------------------------------------------
    # Context block construction
    # ------------------------------------------------------------------

    def _build_context_block(self, chunks: list[Any]) -> str:
        """
        Format retrieved chunks into a numbered context block.
        Enforces token budget by dropping lowest-ranked chunks.
        """
        if not chunks:
            return "No relevant context found."

        formatted_chunks: list[str] = []
        total_chars = 0
        budget_chars = self.max_context_tokens * APPROX_CHARS_PER_TOKEN

        for i, chunk in enumerate(chunks, start=1):
            formatted = self._format_chunk(i, chunk)
            chunk_chars = len(formatted)

            if total_chars + chunk_chars > budget_chars and formatted_chunks:
                logger.debug(
                    "Context budget reached at chunk %d/%d — dropping remaining chunks.",
                    i - 1, len(chunks)
                )
                break

            formatted_chunks.append(formatted)
            total_chars += chunk_chars

        return "\n\n".join(formatted_chunks)

    def _format_chunk(self, index: int, chunk: Any) -> str:
        """
        Format a single RetrievedChunk for inclusion in the prompt.
        Tables get code fences; prose/lists get plain text.
        """
        # Support both dataclass (RetrievedChunk) and dict
        if hasattr(chunk, "content"):
            content      = chunk.content
            chunk_id     = chunk.chunk_id
            doc_id       = chunk.doc_id
            section      = chunk.section_header or "—"
            page         = chunk.page_number
            content_type = chunk.content_type
        else:
            content      = chunk.get("content", "")
            chunk_id     = chunk.get("chunk_id", f"chunk_{index}")
            doc_id       = chunk.get("metadata", {}).get("doc_id", "unknown")
            section      = chunk.get("metadata", {}).get("section_header") or "—"
            page         = chunk.get("metadata", {}).get("page_number", 1)
            content_type = chunk.get("metadata", {}).get("content_type", "prose")

        source_label = (
            f"[CHUNK {index}] chunk_id={chunk_id} | "
            f"doc={doc_id} | section={section} | page={page} | type={content_type}"
        )

        if content_type == "table":
            return f"{source_label}\n```\n{content}\n```"
        else:
            return f"{source_label}\n{content}"

    def _build_user_prompt(self, query: str, context_block: str) -> str:
        return f"""## RETRIEVED CONTEXT

{context_block}

## QUERY

{query}

## INSTRUCTIONS

Answer the query using ONLY the retrieved context above.
- If the context contains the answer, provide it with citations.
- If the context does NOT contain the answer, set is_decline=true.
- Extract any numeric values (rates, fees, timelines) into metric_value.
- Your response must be a single valid JSON object matching the output schema.
"""
