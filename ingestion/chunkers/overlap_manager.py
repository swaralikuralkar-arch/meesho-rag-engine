from __future__ import annotations
import logging
from ingestion.parsers.pdf_layout import ParsedBlock
from ingestion.chunkers.recursive_splitter import count_tokens

logger = logging.getLogger(__name__)

OVERLAP_SENTINEL = "\n\n---context-overlap---\n\n"
_OVERLAPPABLE = frozenset(["prose", "list_item"])

class OverlapManager:
    def __init__(self, overlap_tokens: int = 100, sentinel: str = OVERLAP_SENTINEL):
        self.overlap_tokens = overlap_tokens
        self.sentinel = sentinel

    def apply(self, chunks: list) -> list:
        if not chunks:
            return []
        result, prev = [], None
        for chunk in chunks:
            if not self._is_overlappable(chunk, prev):
                result.append(chunk)
                prev = chunk
                continue
            overlap_text = self._extract_tail(prev.content, self.overlap_tokens)
            if not overlap_text.strip():
                result.append(chunk)
                prev = chunk
                continue
            new_content = overlap_text + self.sentinel + chunk.content
            result.append(ParsedBlock(
                block_type=chunk.block_type,
                content=new_content,
                doc_id=chunk.doc_id,
                page_number=chunk.page_number,
                block_index=chunk.block_index,
                section_header=chunk.section_header,
                table_index=chunk.table_index,
            ))
            prev = chunk
        return result

    def _is_overlappable(self, chunk, prev) -> bool:
        if prev is None: return False
        if chunk.block_type not in _OVERLAPPABLE: return False
        if prev.block_type not in _OVERLAPPABLE: return False
        if chunk.section_header != prev.section_header: return False
        if chunk.doc_id != prev.doc_id: return False
        return True

    def _extract_tail(self, text: str, n_tokens: int) -> str:
        if self.sentinel in text:
            text = text.split(self.sentinel)[-1]
        words = text.split(" ")
        tail_words, token_count = [], 0
        for word in reversed(words):
            wt = count_tokens(word + " ")
            if token_count + wt > n_tokens:
                break
            tail_words.insert(0, word)
            token_count += wt
        return " ".join(tail_words)

def strip_overlap(content: str, sentinel: str = OVERLAP_SENTINEL) -> str:
    if sentinel in content:
        return content.split(sentinel, 1)[1]
    return content
