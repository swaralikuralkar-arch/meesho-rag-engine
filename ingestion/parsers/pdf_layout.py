from __future__ import annotations
import logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import pdfplumber

logger = logging.getLogger(__name__)

try:
    from unstructured.partition.pdf import partition_pdf
    _UNSTRUCTURED_AVAILABLE = True
except ImportError:
    _UNSTRUCTURED_AVAILABLE = False

@dataclass
class ParsedBlock:
    block_type: str
    content: str
    doc_id: str
    page_number: int
    block_index: int
    section_header: str | None = None
    table_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "block_type": self.block_type,
            "content": self.content,
            "metadata": {
                "doc_id": self.doc_id,
                "doc_type": "pdf",
                "page_number": self.page_number,
                "section_header": self.section_header,
                "table_index": self.table_index,
                "block_index": self.block_index,
            },
        }

class TableSerializer:
    @staticmethod
    def serialize(rows: list) -> str:
        if not rows:
            return ""
        cleaned = [
            [(cell or "").replace("|", "\\|").strip().replace("\n", " ") for cell in row]
            for row in rows
        ]
        max_cols = max(len(r) for r in cleaned)
        normalized = [r + [""] * (max_cols - len(r)) for r in cleaned]
        header, body = normalized[0], normalized[1:]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join(["---"] * max_cols) + " |"]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

_HEADING_PATTERN = re.compile(
    r"^(\d+[\.\d]*\s+)?[A-Z][A-Z\s\-\/]{3,}$|^(#{1,4}\s)", re.MULTILINE
)

def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 120 or stripped.endswith("."):
        return False
    return bool(_HEADING_PATTERN.match(stripped))

class MeeshoPDFParser:
    _TABLE_SETTINGS = {
        "vertical_strategy": "lines", "horizontal_strategy": "lines",
        "snap_tolerance": 3, "join_tolerance": 3, "edge_min_length": 3,
        "min_words_vertical": 3, "min_words_horizontal": 1,
        "intersection_tolerance": 3, "text_tolerance": 3,
    }

    def __init__(self, doc_id: str, use_unstructured_fallback: bool = True, min_table_rows: int = 2):
        self.doc_id = doc_id
        self.use_unstructured_fallback = use_unstructured_fallback and _UNSTRUCTURED_AVAILABLE
        self.min_table_rows = min_table_rows

    def parse(self, pdf_path: Path) -> Iterator[ParsedBlock]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        block_idx, table_idx, current_section = 0, 0, None
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables_on_page = page.find_tables(self._TABLE_SETTINGS)
                table_bboxes = [t.bbox for t in tables_on_page]
                for table_obj in tables_on_page:
                    rows = table_obj.extract()
                    if not rows or len(rows) < self.min_table_rows:
                        continue
                    md = TableSerializer.serialize(rows)
                    if not md.strip():
                        continue
                    yield ParsedBlock("table", md, self.doc_id, page_num, block_idx, current_section, table_idx)
                    block_idx += 1
                    table_idx += 1
                for raw_text in self._extract_text_blocks(page, table_bboxes):
                    text = raw_text.strip()
                    if not text:
                        continue
                    if _looks_like_heading(text):
                        current_section = text
                        block_type = "heading"
                    elif text.lstrip().startswith(("*", "-", "*", "-")) or re.match(r"^\d+[\.\)]\s", text):
                        block_type = "list_item"
                    else:
                        block_type = "prose"
                    yield ParsedBlock(block_type, text, self.doc_id, page_num, block_idx, current_section, None)
                    block_idx += 1

    def _extract_text_blocks(self, page, table_bboxes):
        if not table_bboxes:
            return self._split_into_paragraphs(page.extract_text(layout=True) or "")
        text_parts = []
        y_cursor = page.top
        for bb in sorted(table_bboxes, key=lambda b: b[1]):
            if y_cursor < bb[1]:
                crop = page.within_bbox((page.x0, y_cursor, page.x1, bb[1]))
                text_parts.append(crop.extract_text(layout=True) or "")
            y_cursor = max(y_cursor, bb[3])
        if y_cursor < page.bottom:
            crop = page.within_bbox((page.x0, y_cursor, page.x1, page.bottom))
            text_parts.append(crop.extract_text(layout=True) or "")
        return self._split_into_paragraphs("\n\n".join(text_parts))

    @staticmethod
    def _split_into_paragraphs(text: str) -> list:
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def _html_table_to_markdown(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return html
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        return TableSerializer.serialize(rows)
    except Exception:
        return html
