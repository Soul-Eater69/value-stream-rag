"""
Core domain models for the Value Stream RAG system.

These are the canonical data structures used throughout the pipeline.
All models use Pydantic v2 for validation, serialization, and schema generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class ChunkType(str, Enum):
    """Identifies the origin content type of a chunk."""

    SLIDE_TEXT = "slide_text"
    SLIDE_TABLE = "slide_table"
    SLIDE_TITLE = "slide_title"
    SLIDE_NOTES = "slide_notes"
    SECTION = "section"
    SYNTHETIC_SUMMARY = "synthetic_summary"  # LLM-generated slide summary


class EvidenceSource(str, Enum):
    VALUE_STREAM_INDEX = "value_stream_index"
    HISTORICAL_PPT_INDEX = "historical_ppt_index"
    STAGE_INDEX = "stage_index"
    HYBRID = "hybrid"


# ─────────────────────────────────────────────────────────────────────────────
# Value Stream domain models
# ─────────────────────────────────────────────────────────────────────────────


class Stage(BaseModel):
    """A single stage within a Value Stream."""

    id: str = Field(description="Unique stage identifier (e.g. VS01-S02)")
    name: str = Field(description="Human-readable stage name")
    description: str = Field(default="", description="Stage description / purpose")
    sequence_order: int = Field(ge=1, description="Position within the Value Stream")
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageSequence(BaseModel):
    """Ordered list of stages for a Value Stream, with transition logic."""

    value_stream_id: str
    stages: list[Stage] = Field(default_factory=list)
    is_linear: bool = Field(
        default=True,
        description="If False, stages may have conditional branching",
    )

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in sorted(self.stages, key=lambda s: s.sequence_order)]


class ValueStream(BaseModel):
    """
    Represents a single Value Stream as retrieved from Azure AI Search.

    The vector embedding is stored in the index; here we carry the rich metadata
    alongside it so that downstream ranking and explanation can use all fields.
    """

    id: str = Field(description="Unique Value Stream ID from Azure AI Search")
    name: str
    description: str = Field(default="")
    domain: str = Field(default="", description="Business domain / tribe")
    keywords: list[str] = Field(default_factory=list)
    stage_sequence: StageSequence | None = None
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value properties stored in the index",
    )

    # Populated at retrieval time
    similarity_score: float | None = Field(
        default=None, description="Cosine / hybrid score from search"
    )
    rerank_score: float | None = Field(
        default=None, description="Score after cross-encoder reranking"
    )
    final_score: float | None = Field(
        default=None, description="Aggregated score used for ranking"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chunk models
# ─────────────────────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """
    A single text unit extracted from a PPT slide or table.

    Chunks are the atomic unit indexed into vector stores and used for retrieval.
    """

    id: UUID = Field(default_factory=uuid4)
    source_document_id: str = Field(
        description="ID of the parent document (historical card or upload ID)"
    )
    source_document_path: str = Field(description="File path or blob URI")
    slide_index: int = Field(ge=0, description="Zero-based slide index")
    slide_title: str = Field(default="")
    chunk_type: ChunkType
    content: str = Field(description="Raw text content of the chunk")
    token_count: int = Field(ge=0)

    # Table-specific
    table_index: int | None = Field(
        default=None, description="Index of the table on the slide (0-based)"
    )
    table_headers: list[str] = Field(default_factory=list)

    # Enrichment
    section_label: str = Field(
        default="",
        description="High-level section inferred from slide groupings",
    )
    enriched_content: str = Field(
        default="",
        description="Content with injected context (e.g. slide title prepended)",
    )

    # Provenance
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """Return the text that should be embedded (enriched if available)."""
        return self.enriched_content or self.content


class HistoricalIdeaCard(BaseModel):
    """
    Metadata record for a historical idea card PPT with known Value Stream mappings.

    This is stored in the local SQLite database and referenced by chunk metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    file_path: str
    jira_ticket: str | None = Field(default=None, description="JIRA/JITS ticket ID")
    mapped_value_stream_ids: list[str] = Field(
        description="Gold-standard Value Stream IDs from JIRA/JITS"
    )
    mapped_value_stream_names: list[str] = Field(default_factory=list)
    domain: str = Field(default="")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval and recommendation models
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalEvidence(BaseModel):
    """
    A single piece of evidence supporting a Value Stream recommendation.

    Connects a retrieved chunk to the Value Stream it supports, and tracks
    the source so the recommendation can be fully explained.
    """

    chunk_id: str
    chunk_content: str
    chunk_type: ChunkType
    slide_index: int
    slide_title: str
    source_document_id: str
    source_document_path: str

    # Historical card linkage (None for Value Stream index hits)
    historical_card_id: str | None = None
    historical_card_title: str | None = None
    historical_mapped_vs_ids: list[str] = Field(default_factory=list)

    evidence_source: EvidenceSource
    similarity_score: float
    rerank_score: float | None = None


class RecommendationResult(BaseModel):
    """
    Final recommendation output for a single uploaded PPT.

    Contains ranked Value Streams with full evidence chains for explainability.
    """

    recommendation_id: UUID = Field(default_factory=uuid4)
    upload_id: str
    query_summary: str = Field(description="LLM-generated summary of the uploaded PPT")
    recommended_value_streams: list[ValueStream]
    evidence_by_value_stream: dict[str, list[RetrievalEvidence]] = Field(
        default_factory=dict,
        description="Map from VS ID to supporting evidence chunks",
    )
    reasoning: str = Field(
        description="LLM-generated natural language explanation"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_duration_ms: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Upload / runtime models
# ─────────────────────────────────────────────────────────────────────────────


class UploadedPPT(BaseModel):
    """Represents a user-uploaded PPT at runtime."""

    upload_id: str = Field(default_factory=lambda: str(uuid4()))
    original_filename: str
    file_path: str
    file_size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processing_status: str = Field(default="pending")
    chunks: list[Chunk] = Field(default_factory=list)
    recommendation_id: str | None = None
