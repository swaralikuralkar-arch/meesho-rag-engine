from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from ingestion.parsers.pdf_layout import ParsedBlock
from ingestion.parsers.table_extractor import TableTokenBudget
from ingestion.chunkers.recursive_splitter import RecursiveProseChunker, count_tokens
from ingestion.chunkers.overlap_manager import OverlapManager, OVERLAP_SENTINEL, strip_overlap

logger = logging.getLogger(__name__)

@dataclass
class ChunkRecord:
    chunk_id: str
    content: str
    content_with_overlap: str
    doc_id: str
    doc_type: str
    page_number: int
    section_header: str | None
    table_index: int | None
    block_index: int
    chunk_index: int
    total_chunks: int
    content_type: str
    has_overlap: bool
    token_count: int
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "content_with_overlap": self.content_with_overlap,
            "metadata": {
                "doc_id": self.doc_id,
                "doc_type": self.doc_type,
                "page_number": self.page_number,
                "section_header": self.section_header,
                "table_index": self.table_index,
                "block_index": self.block_index,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "content_type": self.content_type,
                "has_overlap": self.has_overlap,
                "token_count": self.token_count,
                "ingested_at": self.ingested_at,
            },
        }

class SemanticChunker:
    def __init__(self, doc_type: str = "pdf", prose_target_tokens: int = 600,
                 prose_max_tokens: int = 800, table_max_tokens: int = 750,
                 overlap_tokens: int = 100):
        self.doc_type = doc_type
        self._prose_chunker = RecursiveProseChunker(target_tokens=prose_target_tokens, max_tokens=prose_max_tokens)
        self._table_budget = TableTokenBudget(max_tokens=table_max_tokens)
        self._overlap_manager = OverlapManager(overlap_tokens=overlap_tokens)

    def chunk(self, blocks: list) -> list:
        split_chunks = []
        for block in blocks:
            split_chunks.extend(self._dispatch(block))
        overlapped_chunks = self._overlap_manager.apply(split_chunks)
        block_chunk_counts, block_chunk_positions = {}, {}
        for chunk in overlapped_chunks:
            key = (chunk.doc_id, chunk.block_index)
            block_chunk_counts[key] = block_chunk_counts.get(key, 0) + 1
        records = []
        for chunk in overlapped_chunks:
            key = (chunk.doc_id, chunk.block_index)
            pos = block_chunk_positions.get(key, 0)
            block_chunk_positions[key] = pos + 1
            has_overlap = OVERLAP_SENTINEL in chunk.content
            clean_content = strip_overlap(chunk.content)
            records.append(ChunkRecord(
                chunk_id=f"{chunk.doc_id}::{chunk.block_index}::{pos}",
                content=clean_content,
                content_with_overlap=chunk.content,
                doc_id=chunk.doc_id,
                doc_type=self.doc_type,
                page_number=chunk.page_number,
                section_header=chunk.section_header,
                table_index=chunk.table_index,
                block_index=chunk.block_index,
                chunk_index=pos,
                total_chunks=block_chunk_counts[key],
                content_type=chunk.block_type,
                has_overlap=has_overlap,
                token_count=count_tokens(clean_content),
            ))
        logger.info("SemanticChunker: %d blocks -> %d chunks", len(blocks), len(records))
        return records

    def _dispatch(self, block: ParsedBlock) -> list:
        if block.block_type == "table":
            return self._table_budget.split(block)
        elif block.block_type in ("prose", "list_item"):
            return self._prose_chunker.split(block)
        elif block.block_type == "heading":
            return [block]
        else:
            return [block]
