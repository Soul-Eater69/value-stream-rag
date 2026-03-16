"""
SQLAlchemy ORM models for the local SQLite persistence layer.

These tables store:
  - historical idea card metadata (ground-truth VS mappings)
  - ingestion job tracking
  - recommendation traces (for evaluation and audit)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Historical idea cards
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalIdeaCardORM(Base):
    __tablename__ = "historical_idea_cards"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    jira_ticket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain: Mapped[str] = mapped_column(String(256), default="")
    mapped_value_stream_ids: Mapped[list] = mapped_column(JSON, default=list)
    mapped_value_stream_names: Mapped[list] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    chunks: Mapped[list[ChunkORM]] = relationship(
        "ChunkORM", back_populates="historical_card", cascade="all, delete-orphan"
    )


class ChunkORM(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("historical_idea_cards.id"), nullable=True
    )
    source_document_path: Mapped[str] = mapped_column(String(1024))
    slide_index: Mapped[int] = mapped_column(Integer)
    slide_title: Mapped[str] = mapped_column(String(512), default="")
    chunk_type: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    enriched_content: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer)
    table_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_label: Mapped[str] = mapped_column(String(256), default="")
    # Versioning: tracks which embedding model produced this chunk's vector.
    # Populated at ingest time from EmbeddingService.model_id.
    # A mismatch between stored model_id and the live service indicates stale
    # embeddings that must be re-ingested before similarity scores are valid.
    embedding_model: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    historical_card: Mapped[HistoricalIdeaCardORM | None] = relationship(
        "HistoricalIdeaCardORM", back_populates="chunks"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion jobs
# ─────────────────────────────────────────────────────────────────────────────


class IngestionJobORM(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_type: Mapped[str] = mapped_column(
        String(64), comment="historical_batch | upload_single"
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    failed_files: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_log: Mapped[list] = mapped_column(JSON, default=list)


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation traces (audit trail)
# ─────────────────────────────────────────────────────────────────────────────


class RecommendationTraceORM(Base):
    __tablename__ = "recommendation_traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    upload_id: Mapped[str] = mapped_column(String(64))
    original_filename: Mapped[str] = mapped_column(String(512))
    query_summary: Mapped[str] = mapped_column(Text, default="")
    recommended_vs_ids: Mapped[list] = mapped_column(JSON, default=list)
    recommended_vs_names: Mapped[list] = mapped_column(JSON, default=list)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    processing_duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    full_result_json: Mapped[dict] = mapped_column(JSON, default=dict)

    # Human feedback (for evaluation loop)
    human_approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    human_corrected_vs_ids: Mapped[list] = mapped_column(JSON, default=list)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB factory
# ─────────────────────────────────────────────────────────────────────────────


def create_db_engine(database_url: str):
    """Create SQLAlchemy engine and initialise schema."""
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(database_url: str):
    engine = create_db_engine(database_url)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
