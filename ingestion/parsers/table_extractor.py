from __future__ import annotations
import logging, re
from ingestion.parsers.pdf_layout import ParsedBlock, TableSerializer

logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    return len(text) // 4 + 1

class TableTokenBudget:
    def __init__(self, max_tokens: int = 750):
        self.max_tokens = max_tokens

    def split(self, block: ParsedBlock) -> list:
        if block.block_type != "table":
            raise ValueError("Only accepts table blocks")
        if count_tokens(block.content) <= self.max_tokens:
            return [block]
        rows = self._parse_markdown_table(block.content)
        if len(rows) < 2:
            return [block]
        header, body = rows[0], rows[1:]
        chunks, current_chunk = [], [header]
        for row in body:
            candidate = current_chunk + [row]
            if count_tokens(TableSerializer.serialize(candidate)) > self.max_tokens and len(current_chunk) > 1:
                chunks.append(current_chunk)
                current_chunk = [header, row]
            else:
                current_chunk = candidate
        if len(current_chunk) > 1:
            chunks.append(current_chunk)
        if not chunks:
            return [block]
        return [
            ParsedBlock(
                block_type="table",
                content=TableSerializer.serialize(chunk_rows),
                doc_id=block.doc_id,
                page_number=block.page_number,
                block_index=block.block_index,
                section_header=block.section_header,
                table_index=block.table_index,
            )
            for chunk_rows in chunks
        ]

    @staticmethod
    def _parse_markdown_table(md: str) -> list:
        rows = []
        for line in md.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if re.match(r"^\|[\s\-\|]+\|$", line):
                continue
            rows.append([c.strip() for c in line.strip("|").split("|")])
        return rows

def extract_tables_from_html(html_str: str) -> list:
    from ingestion.parsers.html_guidelines import HTMLTableExtractor
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_str, "html.parser")
    extractor = HTMLTableExtractor()
    return [md for t in soup.find_all("table") if (md := extractor.extract(t)).strip()]
