"""
LangGraph state definitions.

LangGraph uses TypedDict-based state that is passed between graph nodes.
We define two main state objects:

  1. IngestionState   – used by the historical-PPT ingestion workflow
  2. RecommendationState – used by the runtime recommendation workflow
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.models.domain import (
    Chunk,
    RecommendationResult,
    UploadedPPT,
    ValueStream,
)
from src.search.retriever import RetrievalContext


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion workflow state
# ─────────────────────────────────────────────────────────────────────────────


class IngestionState(TypedDict, total=False):
    """State flowing through the batch ingestion workflow."""

    # Input
    input_records: list[dict]  # Raw IngestionRecord dicts

    # Processing
    current_file: str
    parsed_document: Any  # ParsedDocument
    chunks: list[Chunk]
    embeddings: list[list[float]]
    card_id: str

    # Accumulation
    results: list[dict]  # IngestionResult dicts
    errors: list[str]

    # Control
    job_id: str
    status: str  # pending | running | completed | failed


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation workflow state
# ─────────────────────────────────────────────────────────────────────────────


class RecommendationState(TypedDict, total=False):
    """State flowing through the runtime recommendation workflow."""

    # Input
    upload_id: str
    file_path: str
    original_filename: str
    domain_hint: str | None  # Optional user-provided domain filter

    # Parsing & chunking
    uploaded_ppt: UploadedPPT
    query_chunks: list[Chunk]
    chunk_embeddings: list[list[float]]

    # Query synthesis
    query_summary: str  # LLM-generated summary of the uploaded PPT

    # Retrieval
    retrieval_context: RetrievalContext

    # Ranking
    ranked_value_streams: list[ValueStream]

    # Synthesis
    recommendation: RecommendationResult
    reasoning: str

    # Control flow
    retry_count: int
    errors: list[str]
    status: str  # parsing | retrieving | ranking | synthesising | done | error
