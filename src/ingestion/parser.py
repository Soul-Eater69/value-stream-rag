"""
PPT parsing module using MarkItDown.

Design goals:
  - Extract text and tables per slide, preserving structure.
  - Enrich every slide with its title for downstream context injection.
  - Expose a clean ParsedDocument / ParsedSlide structure consumed by chunking.
  - Handle both historical (batch) and runtime (single-upload) scenarios.

Notes on MarkItDown:
  MarkItDown converts PPTX to Markdown.  Slide boundaries are delimited by
  '<!-- Slide N -->' HTML comments.  Tables are rendered as Markdown tables.
  We post-process this output to reconstruct per-slide structure with explicit
  table detection.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MarkItDown is an optional dependency; guard for environments without it.
try:
    from markitdown import MarkItDown  # type: ignore[import]

    _MARKITDOWN_AVAILABLE = True
except ImportError:
    _MARKITDOWN_AVAILABLE = False
    logger.warning(
        "markitdown not installed.  Install with: pip install 'markitdown[pptx]'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParsedTable:
    """A single Markdown table extracted from a slide."""

    index: int  # 0-based table index on this slide
    raw_markdown: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    plaintext: str = ""  # Flattened table text for embedding


@dataclass
class ParsedSlide:
    """Structured representation of a single slide after parsing."""

    slide_index: int  # 0-based
    title: str
    body_text: str  # All non-table text
    speaker_notes: str
    tables: list[ParsedTable] = field(default_factory=list)
    section_label: str = ""  # e.g. inferred from grouping/title keywords

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def full_text(self) -> str:
        """Combined text including flattened tables."""
        parts = [self.title, self.body_text]
        for t in self.tables:
            parts.append(t.plaintext)
        return "\n".join(p for p in parts if p)


@dataclass
class ParsedDocument:
    """Result of parsing a complete PPTX file."""

    file_path: str
    slide_count: int
    slides: list[ParsedSlide] = field(default_factory=list)
    raw_markdown: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.full_text for s in self.slides)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────


# MarkItDown uses <!-- Slide N --> as a delimiter between slides (1-indexed).
_SLIDE_DELIMITER_RE = re.compile(r"<!--\s*Slide\s+(\d+)\s*-->", re.IGNORECASE)
# A Markdown table row starts with a pipe character.
_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
# Speaker notes are sometimes emitted as a block quote or italics prefix.
_NOTES_RE = re.compile(r"(?:>\s*|_Notes?:_\s*)(.+)$", re.MULTILINE)


class PPTParser:
    """
    Parse a PPTX file into a structured ParsedDocument.

    Strategy
    --------
    1. Run MarkItDown to produce Markdown.
    2. Split on slide-delimiter comments to get per-slide Markdown.
    3. For each slide:
       a. Extract the first H1/H2 line as the title.
       b. Detect contiguous Markdown table blocks and parse them.
       c. Strip table lines to get clean body text.
       d. Extract speaker notes if present.
    4. Optionally infer section labels from title keywords.
    """

    def __init__(self, infer_sections: bool = True) -> None:
        self._infer_sections = infer_sections
        self._md_converter = MarkItDown() if _MARKITDOWN_AVAILABLE else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PPTX file and return a structured ParsedDocument."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PPTX file not found: {file_path}")

        if not _MARKITDOWN_AVAILABLE:
            raise RuntimeError(
                "MarkItDown is required for PPT parsing.  "
                "Install with: pip install 'markitdown[pptx]'"
            )

        logger.info("Parsing PPT: %s", path.name)
        result = self._md_converter.convert(str(path))
        raw_md: str = result.text_content

        slides = self._split_and_parse_slides(raw_md)
        logger.info("Parsed %d slides from %s", len(slides), path.name)

        return ParsedDocument(
            file_path=str(path),
            slide_count=len(slides),
            slides=slides,
            raw_markdown=raw_md,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_and_parse_slides(self, raw_md: str) -> list[ParsedSlide]:
        """Split raw Markdown by slide delimiters and parse each block."""
        # Find all delimiter positions
        boundaries = [
            (m.start(), int(m.group(1))) for m in _SLIDE_DELIMITER_RE.finditer(raw_md)
        ]

        if not boundaries:
            # No delimiters: treat entire doc as a single slide
            return [self._parse_slide_block(raw_md, slide_index=0)]

        slides: list[ParsedSlide] = []
        for i, (start_pos, slide_num) in enumerate(boundaries):
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(raw_md)
            # Skip the delimiter line itself
            block_start = raw_md.index("\n", start_pos) + 1 if "\n" in raw_md[start_pos:end_pos] else start_pos
            slide_md = raw_md[block_start:end_pos].strip()
            slides.append(self._parse_slide_block(slide_md, slide_index=slide_num - 1))

        if self._infer_sections:
            slides = self._annotate_sections(slides)

        return slides

    def _parse_slide_block(self, slide_md: str, slide_index: int) -> ParsedSlide:
        """Parse a single slide's Markdown block."""
        title = self._extract_title(slide_md)
        tables, cleaned_md = self._extract_tables(slide_md)
        body_text = self._clean_body(cleaned_md, title)
        notes = self._extract_notes(slide_md)

        return ParsedSlide(
            slide_index=slide_index,
            title=title,
            body_text=body_text,
            speaker_notes=notes,
            tables=tables,
        )

    def _extract_title(self, slide_md: str) -> str:
        """Extract title from the first H1, H2, or H3 heading."""
        for line in slide_md.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return re.sub(r"^#+\s*", "", stripped).strip()
        # Fall back to first non-empty line
        for line in slide_md.splitlines():
            if line.strip():
                return line.strip()[:120]
        return ""

    def _extract_tables(
        self, slide_md: str
    ) -> tuple[list[ParsedTable], str]:
        """
        Extract Markdown tables from slide content.

        Returns:
            - list of ParsedTable objects
            - slide_md with table lines removed
        """
        lines = slide_md.splitlines()
        tables: list[ParsedTable] = []
        non_table_lines: list[str] = []
        table_buffer: list[str] = []
        table_idx = 0

        for line in lines:
            if _TABLE_ROW_RE.match(line.rstrip()):
                table_buffer.append(line)
            else:
                if table_buffer:
                    parsed = self._parse_markdown_table(
                        "\n".join(table_buffer), table_idx
                    )
                    tables.append(parsed)
                    table_idx += 1
                    table_buffer = []
                non_table_lines.append(line)

        # Flush remaining table buffer
        if table_buffer:
            parsed = self._parse_markdown_table("\n".join(table_buffer), table_idx)
            tables.append(parsed)

        return tables, "\n".join(non_table_lines)

    def _parse_markdown_table(self, raw: str, index: int) -> ParsedTable:
        """Parse a Markdown table string into headers, rows, and plaintext."""
        rows_raw = [
            line for line in raw.splitlines() if "|" in line
        ]
        rows_parsed: list[list[str]] = []
        headers: list[str] = []

        for i, row in enumerate(rows_raw):
            cells = [c.strip() for c in row.strip("|").split("|")]
            # Skip separator row (e.g. |---|---|)
            if all(re.match(r"^-+$", c) for c in cells if c):
                continue
            if i == 0 and not headers:
                headers = cells
            else:
                rows_parsed.append(cells)

        # Build plaintext: "header1: val1 | header2: val2 ..."
        plaintext_parts = []
        if headers:
            plaintext_parts.append(" | ".join(headers))
        for row in rows_parsed:
            if headers:
                cells_text = " | ".join(
                    f"{h}: {v}" for h, v in zip(headers, row, strict=False)
                )
            else:
                cells_text = " | ".join(row)
            plaintext_parts.append(cells_text)

        return ParsedTable(
            index=index,
            raw_markdown=raw,
            headers=headers,
            rows=rows_parsed,
            plaintext="\n".join(plaintext_parts),
        )

    def _clean_body(self, slide_md: str, title: str) -> str:
        """Strip heading, notes markers, and extra whitespace from body text."""
        lines = []
        for line in slide_md.splitlines():
            stripped = line.strip()
            # Skip heading lines
            if stripped.startswith("#"):
                continue
            # Skip note markers
            if re.match(r"^>\s*", stripped) or re.match(r"^_Notes?:_", stripped):
                continue
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        # Collapse 3+ blank lines to 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _extract_notes(self, slide_md: str) -> str:
        """Extract speaker notes if present."""
        matches = _NOTES_RE.findall(slide_md)
        return " ".join(matches).strip()

    def _annotate_sections(self, slides: list[ParsedSlide]) -> list[ParsedSlide]:
        """
        Heuristically assign section labels based on title keywords.

        Common idea-card sections: Problem, Solution, Value, Impact,
        Stakeholders, Milestones, Risks, etc.
        """
        SECTION_KEYWORDS: dict[str, list[str]] = {
            "Problem Statement": ["problem", "challenge", "pain", "issue", "gap"],
            "Proposed Solution": ["solution", "proposal", "approach", "design", "idea"],
            "Business Value": ["value", "benefit", "roi", "impact", "outcome"],
            "Stakeholders": ["stakeholder", "team", "owner", "sponsor", "persona"],
            "Milestones": ["milestone", "timeline", "roadmap", "phase", "schedule"],
            "Risks": ["risk", "concern", "blocker", "dependency", "assumption"],
            "Metrics": ["metric", "kpi", "measure", "success", "criteria"],
        }

        for slide in slides:
            title_lower = slide.title.lower()
            for section, keywords in SECTION_KEYWORDS.items():
                if any(kw in title_lower for kw in keywords):
                    slide.section_label = section
                    break

        return slides
