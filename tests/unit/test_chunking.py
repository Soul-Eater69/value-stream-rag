"""
Unit tests for the chunking strategies.

These tests use synthetic ParsedDocument objects so no real PPTX files
or external services are required.
"""

from __future__ import annotations

import pytest

from src.chunking.chunker import ChunkingConfig, ChunkingMode, PPTChunker
from src.chunking.strategies import (
    HybridHierarchicalStrategy,
    SlideLevelStrategy,
    TableAwareStrategy,
)
from src.ingestion.parser import ParsedDocument, ParsedSlide, ParsedTable
from src.models.domain import ChunkType


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_document() -> ParsedDocument:
    """A minimal ParsedDocument with 2 text-only slides."""
    slides = [
        ParsedSlide(
            slide_index=0,
            title="Problem Statement",
            body_text="We are experiencing a critical gap in our order fulfilment pipeline.\n\n"
                      "This results in delays of up to 3 business days and customer dissatisfaction.",
            speaker_notes="Emphasise the financial impact.",
            tables=[],
            section_label="Problem Statement",
        ),
        ParsedSlide(
            slide_index=1,
            title="Proposed Solution",
            body_text="We propose integrating a real-time inventory system with the ERP.\n\n"
                      "This will reduce fulfilment time by 70%.",
            speaker_notes="",
            tables=[],
            section_label="Proposed Solution",
        ),
    ]
    return ParsedDocument(
        file_path="/fake/idea_card.pptx",
        slide_count=2,
        slides=slides,
    )


@pytest.fixture
def document_with_tables() -> ParsedDocument:
    """A ParsedDocument containing a table slide."""
    table = ParsedTable(
        index=0,
        raw_markdown="| Metric | Current | Target |\n|---|---|---|\n| SLA | 85% | 99% |\n| MTTR | 4h | 30m |",
        headers=["Metric", "Current", "Target"],
        rows=[["SLA", "85%", "99%"], ["MTTR", "4h", "30m"]],
        plaintext="Metric | Current | Target\nMetric: SLA | Current: 85% | Target: 99%\nMetric: MTTR | Current: 4h | Target: 30m",
    )
    slides = [
        ParsedSlide(
            slide_index=0,
            title="KPI Dashboard",
            body_text="The following table shows current vs target KPIs.",
            speaker_notes="",
            tables=[table],
            section_label="Metrics",
        )
    ]
    return ParsedDocument(
        file_path="/fake/kpi_card.pptx",
        slide_count=1,
        slides=slides,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SlideLevelStrategy
# ─────────────────────────────────────────────────────────────────────────────


class TestSlideLevelStrategy:
    def test_produces_one_chunk_per_slide(self, simple_document):
        strategy = SlideLevelStrategy(max_tokens=512, overlap_tokens=64)
        chunks = strategy.chunk(simple_document, "doc-001")
        assert len(chunks) == 2

    def test_chunk_type_is_slide_text(self, simple_document):
        strategy = SlideLevelStrategy()
        chunks = strategy.chunk(simple_document, "doc-001")
        assert all(c.chunk_type == ChunkType.SLIDE_TEXT for c in chunks)

    def test_slide_title_in_enriched_content(self, simple_document):
        strategy = SlideLevelStrategy()
        chunks = strategy.chunk(simple_document, "doc-001")
        assert "Problem Statement" in chunks[0].enriched_content

    def test_source_document_id_set(self, simple_document):
        strategy = SlideLevelStrategy()
        chunks = strategy.chunk(simple_document, "my-doc-id")
        assert all(c.source_document_id == "my-doc-id" for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# TableAwareStrategy
# ─────────────────────────────────────────────────────────────────────────────


class TestTableAwareStrategy:
    def test_table_chunk_created(self, document_with_tables):
        strategy = TableAwareStrategy(max_tokens=512)
        chunks = strategy.chunk(document_with_tables, "doc-002")
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.SLIDE_TABLE]
        assert len(table_chunks) == 1

    def test_table_headers_preserved(self, document_with_tables):
        strategy = TableAwareStrategy(max_tokens=512)
        chunks = strategy.chunk(document_with_tables, "doc-002")
        table_chunk = next(c for c in chunks if c.chunk_type == ChunkType.SLIDE_TABLE)
        assert "Metric" in table_chunk.table_headers

    def test_text_and_table_separate(self, document_with_tables):
        strategy = TableAwareStrategy(max_tokens=512)
        chunks = strategy.chunk(document_with_tables, "doc-002")
        types = {c.chunk_type for c in chunks}
        assert ChunkType.SLIDE_TEXT in types
        assert ChunkType.SLIDE_TABLE in types


# ─────────────────────────────────────────────────────────────────────────────
# HybridHierarchicalStrategy (recommended)
# ─────────────────────────────────────────────────────────────────────────────


class TestHybridHierarchicalStrategy:
    def test_title_chunks_emitted(self, simple_document):
        strategy = HybridHierarchicalStrategy(max_tokens=512)
        chunks = strategy.chunk(simple_document, "doc-003")
        title_chunks = [c for c in chunks if c.chunk_type == ChunkType.SLIDE_TITLE]
        assert len(title_chunks) == 2

    def test_notes_chunk_emitted_when_present(self, simple_document):
        strategy = HybridHierarchicalStrategy(max_tokens=512)
        chunks = strategy.chunk(simple_document, "doc-003")
        notes_chunks = [c for c in chunks if c.chunk_type == ChunkType.SLIDE_NOTES]
        # Only slide 0 has speaker notes
        assert len(notes_chunks) == 1
        assert "financial impact" in notes_chunks[0].content

    def test_table_atomic(self, document_with_tables):
        strategy = HybridHierarchicalStrategy(max_tokens=512)
        chunks = strategy.chunk(document_with_tables, "doc-004")
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.SLIDE_TABLE]
        assert len(table_chunks) == 1
        # Ensure table content is present
        assert "SLA" in table_chunks[0].content or "Metric" in table_chunks[0].content

    def test_enriched_content_has_section_label(self, simple_document):
        strategy = HybridHierarchicalStrategy(max_tokens=512)
        chunks = strategy.chunk(simple_document, "doc-005")
        for chunk in chunks:
            if chunk.section_label:
                assert chunk.section_label in chunk.enriched_content


# ─────────────────────────────────────────────────────────────────────────────
# PPTChunker facade
# ─────────────────────────────────────────────────────────────────────────────


class TestPPTChunker:
    def test_default_mode_is_hybrid(self):
        chunker = PPTChunker()
        assert chunker.config.mode == ChunkingMode.HYBRID_HIERARCHICAL

    def test_custom_mode(self):
        cfg = ChunkingConfig(mode=ChunkingMode.TABLE_AWARE)
        chunker = PPTChunker(cfg)
        assert chunker.config.mode == ChunkingMode.TABLE_AWARE

    def test_chunk_document_returns_list(self, simple_document):
        chunker = PPTChunker()
        chunks = chunker.chunk_document(simple_document, "doc-xyz")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_token_count_set(self, simple_document):
        chunker = PPTChunker()
        chunks = chunker.chunk_document(simple_document, "doc-xyz")
        assert all(c.token_count >= 0 for c in chunks)
