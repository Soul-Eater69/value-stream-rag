"""
PPTChunker: orchestrates chunking strategy selection and execution.

Usage:
    chunker = PPTChunker(config=ChunkingConfig())
    chunks = chunker.chunk_document(parsed_doc, source_document_id="card-001")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from src.ingestion.parser import ParsedDocument
from src.models.domain import Chunk

from .strategies import (
    BaseChunkingStrategy,
    HybridHierarchicalStrategy,
    SemanticStrategy,
    SlideLevelStrategy,
    TableAwareStrategy,
)

logger = logging.getLogger(__name__)


class ChunkingMode(str, Enum):
    SLIDE_LEVEL = "slide_level"
    TABLE_AWARE = "table_aware"
    SEMANTIC = "semantic"
    HYBRID_HIERARCHICAL = "hybrid_hierarchical"  # recommended


@dataclass
class ChunkingConfig:
    mode: ChunkingMode = ChunkingMode.HYBRID_HIERARCHICAL
    max_tokens: int = 512
    overlap_tokens: int = 64


class PPTChunker:
    """Facade that selects and executes the appropriate chunking strategy."""

    _STRATEGY_MAP: dict[ChunkingMode, type[BaseChunkingStrategy]] = {
        ChunkingMode.SLIDE_LEVEL: SlideLevelStrategy,
        ChunkingMode.TABLE_AWARE: TableAwareStrategy,
        ChunkingMode.SEMANTIC: SemanticStrategy,
        ChunkingMode.HYBRID_HIERARCHICAL: HybridHierarchicalStrategy,
    }

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        strategy_cls = self._STRATEGY_MAP[self.config.mode]
        self._strategy: BaseChunkingStrategy = strategy_cls(
            max_tokens=self.config.max_tokens,
            overlap_tokens=self.config.overlap_tokens,
        )
        logger.info(
            "PPTChunker initialised with strategy=%s max_tokens=%d",
            self.config.mode,
            self.config.max_tokens,
        )

    def chunk_document(
        self, document: ParsedDocument, source_document_id: str
    ) -> list[Chunk]:
        """Chunk a ParsedDocument and return a list of Chunk objects."""
        chunks = self._strategy.chunk(document, source_document_id)
        logger.info(
            "Produced %d chunks from %s (doc_id=%s)",
            len(chunks),
            document.file_path,
            source_document_id,
        )
        return chunks
