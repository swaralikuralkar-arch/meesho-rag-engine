"""
core/generation/structured_output.py
=======================================
Enforces structured JSON output from the LLM and validates it against
the RAGResponse Pydantic schema.

Failure modes handled:
  1. LLM returns malformed JSON       → retry with explicit repair prompt
  2. JSON parses but fails Pydantic   → retry with schema error injected
  3. All retries exhausted            → return a structured DECLINE response
                                        (never raise to the caller)

Retry strategy (tenacity):
  - Max 3 attempts total
  - Exponential backoff: 1s, 2s (for API rate limit headroom)
  - On each retry, the previous error is injected into the prompt so
    the LLM can self-correct rather than repeating the same mistake.

LLM backend:
  - Primary: OpenAI-compatible API (works with OpenAI, Anthropic via
    openai-compat, local vLLM, Ollama, etc.)
  - Model: configurable via MEESHO_LLM_MODEL env var
  - Default: "gpt-4o-mini" (fast, cheap, strong JSON compliance)

Public API:
  StructuredOutputParser.parse(query, prompt_pair, retrieved_chunks)
      → RAGResponse
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIError, RateLimitError

from core.schemas import RAGResponse, Citation, DeclineReason, LLMRawOutput

logger = logging.getLogger(__name__)

DEFAULT_MODEL    = os.environ.get("MEESHO_LLM_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.environ.get("MEESHO_LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
MAX_TOKENS       = 1500


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from a text string.
    Handles cases where the LLM wraps JSON in markdown code fences.
    """
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")

    # Walk forward to find the matching closing brace
    depth = 0
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError("Malformed JSON: unmatched braces in LLM response")


# ---------------------------------------------------------------------------
# Structured output parser
# ---------------------------------------------------------------------------

class StructuredOutputParser:
    """
    Calls the LLM, parses the JSON response, and validates it against
    the RAGResponse schema with automatic retry on failure.

    Parameters
    ----------
    model : str
        LLM model name. Default: gpt-4o-mini.
    base_url : str
        OpenAI-compatible API base URL.
    api_key : str
        API key. Falls back to OPENAI_API_KEY env var.
    temperature : float
        LLM temperature. Use 0.0 for deterministic JSON output.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key or "placeholder",
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        query: str,
        prompt_pair,          # PromptPair from prompt_builder.py
        retrieved_chunks: list[Any],
    ) -> RAGResponse:
        """
        Call the LLM and return a validated RAGResponse.
        Never raises — returns a decline on unrecoverable failure.

        Parameters
        ----------
        query : str
            Original user query (for decline response attribution).
        prompt_pair : PromptPair
            System + user prompts from PromptBuilder.
        retrieved_chunks : list[RetrievedChunk]
            The chunks passed to the prompt (used for citation validation).

        Returns
        -------
        RAGResponse
            Always returns a valid RAGResponse, even on LLM failure.
        """
        messages = prompt_pair.to_messages()

        try:
            raw_json = self._call_with_retry(messages)
            response = self._parse_and_validate(query, raw_json, retrieved_chunks)
            return response

        except Exception as exc:
            logger.error(
                "StructuredOutputParser failed after all retries: %s", exc
            )
            return RAGResponse.make_decline(
                query=query,
                reason=DeclineReason.NO_CONTEXT,
                message=(
                    "An error occurred while generating the response. "
                    "Please try again or contact Meesho support."
                ),
            )

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((APIError, RateLimitError, ValueError)),
        reraise=True,
    )
    def _call_with_retry(self, messages: list[dict]) -> str:
        """
        Call the LLM API with exponential backoff retry.
        Returns the raw response text.
        """
        logger.debug("Calling LLM: model=%s, messages=%d", self.model, len(messages))

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},  # enforces JSON mode
        )

        raw = response.choices[0].message.content
        logger.debug("LLM raw response: %s", raw[:200] if raw else "EMPTY")

        if not raw or not raw.strip():
            raise ValueError("LLM returned empty response")

        return raw

    # ------------------------------------------------------------------
    # Parse and validate
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        query: str,
        raw_json: str,
        retrieved_chunks: list[Any],
    ) -> RAGResponse:
        """
        Parse raw JSON string into a validated RAGResponse.
        Injects query and maps chunk metadata into citations.
        """
        # Step 1: Extract JSON object from raw text
        json_str = _extract_json(raw_json)

        # Step 2: Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error: {exc}  Raw: {json_str[:300]!r}")

        # Step 3: Inject query (LLM doesn't generate this field)
        data["query"] = query

        # Step 4: Enrich citations with metadata from retrieved chunks
        data["citations"] = self._enrich_citations(
            data.get("citations", []),
            retrieved_chunks,
        )

        # Step 5: Validate against RAGResponse schema
        try:
            return RAGResponse.model_validate(data)
        except Exception as exc:
            raise ValueError(f"RAGResponse validation failed: {exc}")

    def _enrich_citations(
        self,
        raw_citations: list[dict],
        retrieved_chunks: list[Any],
    ) -> list[dict]:
        """
        The LLM may omit some metadata fields from citations (page_number,
        content_type etc). Backfill them from the retrieved chunks using
        chunk_id as the join key.
        """
        # Build lookup: chunk_id → chunk data
        chunk_lookup: dict[str, Any] = {}
        for chunk in retrieved_chunks:
            if hasattr(chunk, "chunk_id"):
                chunk_lookup[chunk.chunk_id] = chunk
            elif isinstance(chunk, dict):
                chunk_lookup[chunk.get("chunk_id", "")] = chunk

        enriched = []
        for cite in raw_citations:
            cid = cite.get("chunk_id", "")
            chunk = chunk_lookup.get(cid)

            if chunk is None:
                logger.warning(
                    "Citation chunk_id %r not found in retrieved chunks — skipping.", cid
                )
                continue

            # Backfill missing fields from the retrieved chunk
            if hasattr(chunk, "doc_id"):
                cite.setdefault("doc_id",         chunk.doc_id)
                cite.setdefault("section_header", chunk.section_header)
                cite.setdefault("page_number",    chunk.page_number)
                cite.setdefault("content_type",   chunk.content_type)
            elif isinstance(chunk, dict):
                meta = chunk.get("metadata", {})
                cite.setdefault("doc_id",         meta.get("doc_id", "unknown"))
                cite.setdefault("section_header", meta.get("section_header"))
                cite.setdefault("page_number",    meta.get("page_number", 1))
                cite.setdefault("content_type",   meta.get("content_type", "prose"))

            enriched.append(cite)

        return enriched
