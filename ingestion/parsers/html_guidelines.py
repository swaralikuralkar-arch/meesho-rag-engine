"""
ingestion/parsers/html_guidelines.py
"""
from __future__ import annotations
import logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from bs4 import BeautifulSoup, NavigableString, Tag
from ingestion.parsers.pdf_layout import ParsedBlock, TableSerializer

logger = logging.getLogger(__name__)
_SKIP_TAGS = frozenset(["script","style","noscript","nav","footer","header","aside","form","button","svg","meta"])
_HEADING_TAGS = frozenset(["h1","h2","h3","h4","h5","h6"])
_PROSE_TAGS = frozenset(["p","div","span","section","article","main"])

class HTMLTableExtractor:
    def extract(self, table_tag: Tag) -> str:
        grid = self._build_grid(table_tag)
        return TableSerializer.serialize(grid) if grid else ""

    def _build_grid(self, table_tag: Tag) -> list[list[str]]:
        raw_rows, header_row_indices = [], set()
        def _add_rows(section, mark_as_header):
            for tr in section.find_all("tr", recursive=False):
                idx = len(raw_rows)
                raw_rows.append(tr.find_all(["td","th"], recursive=False))
                if mark_as_header:
                    header_row_indices.add(idx)
        thead = table_tag.find("thead")
        tbody = table_tag.find("tbody") or table_tag
        tfoot = table_tag.find("tfoot")
        if thead: _add_rows(thead, True)
        _add_rows(tbody, False)
        if tfoot: _add_rows(tfoot, False)
        if not raw_rows: return []
        grid = {}
        for row_idx, cells in enumerate(raw_rows):
            col_cursor = 0
            for cell in cells:
                while (row_idx, col_cursor) in grid: col_cursor += 1
                text = cell.get_text(separator=" ", strip=True).replace("|","\\|")
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                for dr in range(rowspan):
                    for dc in range(colspan):
                        grid[(row_idx+dr, col_cursor+dc)] = text
                col_cursor += colspan
        if not grid: return []
        n_rows = max(r for r,_ in grid)+1
        n_cols = max(c for _,c in grid)+1
        result = [[grid.get((r,c),"") for c in range(n_cols)] for r in range(n_rows)]
        return result

class MeeshoHTMLParser:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self._table_extractor = HTMLTableExtractor()

    def parse_file(self, html_path: Path) -> Iterator[ParsedBlock]:
        raw = Path(html_path).read_text(encoding="utf-8", errors="replace")
        yield from self.parse_string(raw)

    def parse_string(self, raw_html: str) -> Iterator[ParsedBlock]:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup.find_all(_SKIP_TAGS): tag.decompose()
        body = soup.find("main") or soup.find("article") or soup.find("body") or soup
        block_idx, table_idx, current_section = 0, 0, None
        for element in body.descendants:
            if not isinstance(element, Tag): continue
            if element.name in _SKIP_TAGS: continue
            if not self._is_top_level_block(element): continue
            tag = element.name
            if tag in _HEADING_TAGS:
                text = element.get_text(separator=" ", strip=True)
                if not text: continue
                current_section = text
                yield ParsedBlock("heading", f"{'#'*int(tag[1])} {text}", self.doc_id, 1, block_idx, current_section)
                block_idx += 1
            elif tag == "table":
                md_table = self._table_extractor.extract(element)
                if not md_table.strip(): continue
                yield ParsedBlock("table", md_table, self.doc_id, 1, block_idx, current_section, table_idx)
                block_idx += 1; table_idx += 1
            elif tag in ("ul","ol"):
                items = self._extract_list(element)
                if not items: continue
                yield ParsedBlock("list_item", items, self.doc_id, 1, block_idx, current_section)
                block_idx += 1
            elif tag in _PROSE_TAGS:
                if element.find(["p","div","table","ul","ol","h1","h2","h3","h4","h5","h6"]): continue
                text = element.get_text(separator=" ", strip=True)
                if len(text) < 20: continue
                yield ParsedBlock("prose", text, self.doc_id, 1, block_idx, current_section)
                block_idx += 1

    @staticmethod
    def _is_top_level_block(tag: Tag) -> bool:
        _TOP_LEVEL = _HEADING_TAGS | {"table","ul","ol","p"}
        if tag.name not in _TOP_LEVEL and tag.name not in _PROSE_TAGS: return False
        for ancestor in tag.parents:
            if not isinstance(ancestor, Tag): continue
            if ancestor.name in _TOP_LEVEL and ancestor.name != tag.name: return False
        return True

    @staticmethod
    def _extract_list(list_tag: Tag) -> str:
        lines, ordered = [], list_tag.name == "ol"
        def _recurse(node, depth, counter_start):
            for i, li in enumerate(node.find_all("li", recursive=False), start=counter_start):
                prefix = "  " * depth
                bullet = f"{i}." if ordered else "-"
                nested = li.find(["ul","ol"])
                text = ("".join(c for c in li.children if isinstance(c, NavigableString)).strip()
                        if nested else li.get_text(separator=" ", strip=True))
                lines.append(f"{prefix}{bullet} {text}")
                if nested: _recurse(nested, depth+1, 1)
        _recurse(list_tag, 0, 1)
        return "\n".join(lines)
