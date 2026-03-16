"""
Pluggable chunking strategies.

Architecture decision: We use the Strategy pattern so the chunking algorithm
can be swapped without touching the rest of the pipeline.

Recommended strategy for this system: HybridHierarchicalStrategy
  - Slide-level parent chunks preserve context.
  - Sub-chunks at text-block and table granularity are used for retrieval.
  - Tables are always treated as atomic units.
  - Title is always prepended for context injection.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from uuid import uuid4

import tiktoken

from src.ingestion.parser import ParsedDocument, ParsedSlide, ParsedTable
from src.models.domain import Chunk, ChunkType

logger = logging.getLogger(__name__)

# Tokenizer – use cl100k_base (same as text-embedding-3-*)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class BaseChunkingStrategy(ABC):
    """Base class for all chunking strategies."""

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @abstractmethod
    def chunk(self, document: ParsedDocument, source_document_id: str) -> list[Chunk]:
        """Convert a ParsedDocument into a list of Chunks."""
        ...

    # Shared helpers -----------------------------------------------------------

    def _make_chunk(
        self,
        *,
        source_document_id: str,
        source_document_path: str,
        slide: ParsedSlide,
        chunk_type: ChunkType,
        content: str,
        table_index: int | None = None,
        table_headers: list[str] | None = None,
    ) -> Chunk:
        """Factory method that also injects slide title for context."""
        enriched = self._enrich_content(content, slide.title, slide.section_label)
        return Chunk(
            id=uuid4(),
            source_document_id=source_document_id,
            source_document_path=source_document_path,
            slide_index=slide.slide_index,
            slide_title=slide.title,
            chunk_type=chunk_type,
            content=content,
            token_count=_count_tokens(content),
            enriched_content=enriched,
            section_label=slide.section_label,
            table_index=table_index,
            table_headers=table_headers or [],
        )

    @staticmethod
    def _enrich_content(content: str, title: str, section: str) -> str:
        """Prepend slide title and section to content for embedding."""
        prefix_parts = []
        if section:
            prefix_parts.append(f"[Section: {section}]")
        if title:
            prefix_parts.append(f"Slide: {title}")
        prefix = " | ".join(prefix_parts)
        return f"{prefix}\n{content}" if prefix else content

    def _split_long_text(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        """Split a long text block into overlapping token windows."""
        tokens = _TOKENIZER.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks: list[str] = []
        step = max_tokens - overlap_tokens
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start : start + max_tokens]
            chunks.append(_TOKENIZER.decode(chunk_tokens))
            if start + max_tokens >= len(tokens):
                break
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Slide-level (coarse, simple)
# ─────────────────────────────────────────────────────────────────────────────


class SlideLevelStrategy(BaseChunkingStrategy):
    """
    One chunk per slide, all content combined.

    Suitable for: short slides with minimal content, quick PoC.
    Disadvantage: tables and text mixed together, long slides may exceed limits.
    """

    def chunk(self, document: ParsedDocument, source_document_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for slide in document.slides:
            full = slide.full_text
            if not full.strip():
                continue
            # Possibly split if very long
            for part in self._split_long_text(full, self.max_tokens, self.overlap_tokens):
                chunks.append(
                    self._make_chunk(
                        source_document_id=source_document_id,
                        source_document_path=document.file_path,
                        slide=slide,
                        chunk_type=ChunkType.SLIDE_TEXT,
                        content=part,
                    )
                )
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Table-aware (text + tables separated)
# ─────────────────────────────────────────────────────────────────────────────


class TableAwareStrategy(BaseChunkingStrategy):
    """
    Separate chunks for body text and for each table.

    This ensures table content is always embedded atomically.
    Tables are never split mid-row.
    """

    def chunk(self, document: ParsedDocument, source_document_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for slide in document.slides:
            # Text chunks
            if slide.body_text.strip():
                for part in self._split_long_text(
                    slide.body_text, self.max_tokens, self.overlap_tokens
                ):
                    chunks.append(
                        self._make_chunk(
                            source_document_id=source_document_id,
                            source_document_path=document.file_path,
                            slide=slide,
                            chunk_type=ChunkType.SLIDE_TEXT,
                            content=part,
                        )
                    )

            # Table chunks (one per table, atomic)
            for table in slide.tables:
                table_text = table.plaintext or table.raw_markdown
                if not table_text.strip():
                    continue
                # Tables that exceed the token limit: embed the full table
                # but flag it – we do NOT split tables across chunks.
                truncated = _truncate_to_tokens(table_text, self.max_tokens)
                chunks.append(
                    self._make_chunk(
                        source_document_id=source_document_id,
                        source_document_path=document.file_path,
                        slide=slide,
                        chunk_type=ChunkType.SLIDE_TABLE,
                        content=truncated,
                        table_index=table.index,
                        table_headers=table.headers,
                    )
                )

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Semantic chunking (LLM-boundary detection)
# ─────────────────────────────────────────────────────────────────────────────


class SemanticStrategy(BaseChunkingStrategy):
    """
    Chunk at semantic boundaries using sentence-level splitting.

    Falls back to TokenAware splitting for tables.
    Note: Heavier compute – suitable for offline ingestion, not runtime.
    """

    def chunk(self, document: ParsedDocument, source_document_id: str) -> list[Chunk]:
        # Import here to keep dependency optional
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.max_tokens,
                chunk_overlap=self.overlap_tokens,
            )
        except ImportError:
            logger.warning(
                "langchain_text_splitters not available, falling back to slide-level."
            )
            return SlideLevelStrategy(self.max_tokens, self.overlap_tokens).chunk(
                document, source_document_id
            )

        chunks: list[Chunk] = []
        for slide in document.slides:
            if slide.body_text.strip():
                parts = splitter.split_text(slide.body_text)
                for part in parts:
                    chunks.append(
                        self._make_chunk(
                            source_document_id=source_document_id,
                            source_document_path=document.file_path,
                            slide=slide,
                            chunk_type=ChunkType.SLIDE_TEXT,
                            content=part,
                        )
                    )

            for table in slide.tables:
                table_text = table.plaintext or table.raw_markdown
                if table_text.strip():
                    chunks.append(
                        self._make_chunk(
                            source_document_id=source_document_id,
                            source_document_path=document.file_path,
                            slide=slide,
                            chunk_type=ChunkType.SLIDE_TABLE,
                            content=_truncate_to_tokens(table_text, self.max_tokens),
                            table_index=table.index,
                            table_headers=table.headers,
                        )
                    )

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Hybrid Hierarchical (RECOMMENDED)
# ─────────────────────────────────────────────────────────────────────────────


class HybridHierarchicalStrategy(BaseChunkingStrategy):
    """
    Recommended strategy for this system.

    Produces three chunk levels:
      1. Slide-level summary chunk   (parent context, not indexed separately)
      2. Body text sub-chunks        (split at paragraph/sentence boundaries)
      3. Table chunks                (one per table, atomic, never split)

    All sub-chunks carry their parent slide title in enriched_content
    to preserve context for embedding.

    Advantages:
      - Tables are always atomic → accurate table retrieval.
      - Text sub-chunks are small enough for precise semantic matching.
      - Slide title context is injected into every sub-chunk.
      - Section labels allow metadata filtering at retrieval time.

    For historical (offline) ingestion: all three levels are indexed.
    For runtime uploads: only sub-chunks are embedded (level 2 + 3).
    """

    def chunk(self, document: ParsedDocument, source_document_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for slide in document.slides:
            chunks.extend(
                self._process_slide(slide, document.file_path, source_document_id)
            )
        return chunks

    def _process_slide(
        self, slide: ParsedSlide, file_path: str, source_document_id: str
    ) -> list[Chunk]:
        slide_chunks: list[Chunk] = []

        # ── Level 1: Slide title chunk (always emit, even for slides with only tables)
        if slide.title.strip():
            slide_chunks.append(
                self._make_chunk(
                    source_document_id=source_document_id,
                    source_document_path=file_path,
                    slide=slide,
                    chunk_type=ChunkType.SLIDE_TITLE,
                    content=slide.title,
                )
            )

        # ── Level 2: Body text sub-chunks
        if slide.body_text.strip():
            paragraphs = self._split_into_paragraphs(slide.body_text)
            buffer = ""
            for para in paragraphs:
                candidate = f"{buffer}\n{para}".strip() if buffer else para
                if _count_tokens(candidate) <= self.max_tokens:
                    buffer = candidate
                else:
                    if buffer:
                        slide_chunks.append(
                            self._make_chunk(
                                source_document_id=source_document_id,
                                source_document_path=file_path,
                                slide=slide,
                                chunk_type=ChunkType.SLIDE_TEXT,
                                content=buffer.strip(),
                            )
                        )
                    buffer = para
            if buffer:
                slide_chunks.append(
                    self._make_chunk(
                        source_document_id=source_document_id,
                        source_document_path=file_path,
                        slide=slide,
                        chunk_type=ChunkType.SLIDE_TEXT,
                        content=buffer.strip(),
                    )
                )

        # ── Level 3: Table sub-chunks (atomic – never split)
        for table in slide.tables:
            table_text = table.plaintext or table.raw_markdown
            if not table_text.strip():
                continue
            slide_chunks.append(
                self._make_chunk(
                    source_document_id=source_document_id,
                    source_document_path=file_path,
                    slide=slide,
                    chunk_type=ChunkType.SLIDE_TABLE,
                    content=_truncate_to_tokens(table_text, self.max_tokens),
                    table_index=table.index,
                    table_headers=table.headers,
                )
            )

        # ── Speaker notes (emit as separate chunk if present)
        if slide.speaker_notes.strip():
            slide_chunks.append(
                self._make_chunk(
                    source_document_id=source_document_id,
                    source_document_path=file_path,
                    slide=slide,
                    chunk_type=ChunkType.SLIDE_NOTES,
                    content=_truncate_to_tokens(
                        slide.speaker_notes, self.max_tokens // 2
                    ),
                )
            )

        return slide_chunks

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        """Split on blank lines, falling back to sentence boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        # Further split paragraphs that are single very long sentences
        result: list[str] = []
        for para in paragraphs:
            if _count_tokens(para) > 256:
                # Sentence-level split
                sentences = [s.strip() + "." for s in para.split(". ") if s.strip()]
                result.extend(sentences)
            else:
                result.append(para)
        return result
