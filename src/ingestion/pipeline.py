"""
IngestionPipeline: orchestrates parsing → chunking → embedding → indexing
for historical idea-card PPTs.

This pipeline is designed for offline batch ingestion of historical PPTs
that have known Value Stream mappings from JIRA/JITS.

Steps
-----
1. Parse each PPTX with PPTParser.
2. Chunk with PPTChunker (HybridHierarchical by default).
3. Embed each chunk using the configured EmbeddingService.
4. Upsert embeddings + metadata into ChromaDB (local) and optionally
   Azure AI Search (cloud).
5. Persist chunk metadata and idea-card records in SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from src.chunking.chunker import ChunkingConfig, PPTChunker
from src.ingestion.parser import PPTParser
from src.models.domain import Chunk, HistoricalIdeaCard

logger = logging.getLogger(__name__)


@dataclass
class IngestionRecord:
    """Input specification for a single historical PPT."""

    file_path: str
    mapped_value_stream_ids: list[str]
    mapped_value_stream_names: list[str] = field(default_factory=list)
    jira_ticket: str | None = None
    title: str = ""
    domain: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result summary for a single PPT ingestion."""

    card_id: str
    file_path: str
    slide_count: int
    chunk_count: int
    success: bool
    error: str | None = None


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for historical idea-card PPTs.

    Dependencies are injected to keep this class testable.
    """

    def __init__(
        self,
        embedding_service,  # EmbeddingService (injected)
        vector_store,        # VectorStore (injected)
        db_session_factory,  # SQLAlchemy session factory (injected)
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._embedder = embedding_service
        self._vector_store = vector_store
        self._session_factory = db_session_factory
        self._parser = PPTParser()
        self._chunker = PPTChunker(chunking_config or ChunkingConfig())

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------

    def ingest_batch(self, records: list[IngestionRecord]) -> list[IngestionResult]:
        """Ingest a list of historical PPTs. Returns per-file results."""
        results = []
        for record in records:
            result = self.ingest_single(record)
            results.append(result)
            if result.success:
                logger.info("Ingested %s → %d chunks", record.file_path, result.chunk_count)
            else:
                logger.error("Failed to ingest %s: %s", record.file_path, result.error)
        return results

    def ingest_single(self, record: IngestionRecord) -> IngestionResult:
        """Ingest a single historical PPT."""
        card_id = str(uuid4())
        try:
            # 1. Parse
            parsed = self._parser.parse(record.file_path)

            # 2. Build idea-card domain object
            card = HistoricalIdeaCard(
                id=card_id,
                title=record.title or Path(record.file_path).stem,
                file_path=record.file_path,
                jira_ticket=record.jira_ticket,
                mapped_value_stream_ids=record.mapped_value_stream_ids,
                mapped_value_stream_names=record.mapped_value_stream_names,
                domain=record.domain,
                metadata=record.metadata,
            )

            # 3. Chunk
            chunks = self._chunker.chunk_document(parsed, card_id)
            card.chunk_count = len(chunks)

            # 4. Embed
            texts = [c.embedding_text for c in chunks]
            embeddings = self._embedder.embed_batch(texts)

            # 5. Upsert into vector store with metadata
            self._vector_store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                mapped_value_stream_ids=record.mapped_value_stream_ids,
            )

            # 6. Persist to SQLite
            self._persist_to_db(card, chunks)

            return IngestionResult(
                card_id=card_id,
                file_path=record.file_path,
                slide_count=parsed.slide_count,
                chunk_count=len(chunks),
                success=True,
            )

        except Exception as exc:
            logger.exception("Ingestion failed for %s", record.file_path)
            return IngestionResult(
                card_id=card_id,
                file_path=record.file_path,
                slide_count=0,
                chunk_count=0,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_to_db(self, card: HistoricalIdeaCard, chunks: list[Chunk]) -> None:
        """Persist idea card and chunk records to SQLite."""
        from src.models.database import ChunkORM, HistoricalIdeaCardORM

        session = self._session_factory()
        try:
            card_orm = HistoricalIdeaCardORM(
                id=card.id,
                title=card.title,
                file_path=card.file_path,
                jira_ticket=card.jira_ticket,
                domain=card.domain,
                mapped_value_stream_ids=card.mapped_value_stream_ids,
                mapped_value_stream_names=card.mapped_value_stream_names,
                chunk_count=card.chunk_count,
            )
            session.add(card_orm)

            for chunk in chunks:
                chunk_orm = ChunkORM(
                    id=str(chunk.id),
                    source_document_id=chunk.source_document_id,
                    source_document_path=chunk.source_document_path,
                    slide_index=chunk.slide_index,
                    slide_title=chunk.slide_title,
                    chunk_type=chunk.chunk_type.value,
                    content=chunk.content,
                    enriched_content=chunk.enriched_content,
                    token_count=chunk.token_count,
                    table_index=chunk.table_index,
                    section_label=chunk.section_label,
                )
                session.add(chunk_orm)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
