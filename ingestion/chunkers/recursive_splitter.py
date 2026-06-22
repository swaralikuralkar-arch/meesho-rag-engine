from __future__ import annotations
import logging, re
from ingestion.parsers.pdf_layout import ParsedBlock
import tiktoken

logger = logging.getLogger(__name__)

def _load_tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning("tiktoken failed (%s). Using char/4 fallback.", exc)
        return None

_TOKENIZER = _load_tokenizer()

def count_tokens(text: str) -> int:
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    return len(text) // 4 + 1

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]

class RecursiveProseChunker:
    def __init__(self, target_tokens: int = 600, max_tokens: int = 800, separators=None):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.separators = separators or _SEPARATORS

    def split(self, block: ParsedBlock) -> list:
        if block.block_type not in ("prose", "list_item", "heading"):
            raise ValueError(f"Cannot split block_type: {block.block_type!r}")
        if count_tokens(block.content) <= self.max_tokens:
            return [block]
        raw_chunks = self._recursive_split(block.content)
        result = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            result.append(ParsedBlock(
                block_type=block.block_type,
                content=chunk_text,
                doc_id=block.doc_id,
                page_number=block.page_number,
                block_index=block.block_index,
                section_header=block.section_header,
                table_index=None,
            ))
        return result if result else [block]

    def _recursive_split(self, text: str) -> list:
        if count_tokens(text) <= self.max_tokens:
            return [text]
        for sep in self.separators:
            if sep not in text:
                continue
            parts = text.split(sep)
            if len(parts) < 2:
                continue
            chunks = self._merge_parts(parts, sep)
            result = []
            for chunk in chunks:
                if count_tokens(chunk) > self.max_tokens:
                    result.extend(self._recursive_split(chunk))
                else:
                    result.append(chunk)
            return result
        return self._hard_token_split(text)

    def _merge_parts(self, parts: list, sep: str) -> list:
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if count_tokens(candidate) <= self.target_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def _hard_token_split(self, text: str) -> list:
        words = text.split(" ")
        chunks, current_words, current_tokens = [], [], 0
        for word in words:
            wt = count_tokens(word + " ")
            if current_tokens + wt > self.target_tokens and current_words:
                chunks.append(" ".join(current_words))
                current_words, current_tokens = [word], wt
            else:
                current_words.append(word)
                current_tokens += wt
        if current_words:
            chunks.append(" ".join(current_words))
        return chunks
