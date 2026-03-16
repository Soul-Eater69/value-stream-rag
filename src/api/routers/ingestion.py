"""
Ingestion management endpoints.

POST /api/v1/ingestion/historical/batch
  Trigger batch ingestion of historical idea-card PPTs.

GET /api/v1/ingestion/status/{job_id}
  Check ingestion job status.

GET /api/v1/ingestion/stats
  Return index statistics (chunk count, card count, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.api.dependencies import get_ingestion_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────


class IngestionRecordIn(BaseModel):
    file_path: str
    mapped_value_stream_ids: list[str]
    mapped_value_stream_names: list[str] = []
    jira_ticket: str | None = None
    title: str = ""
    domain: str = ""


class BatchIngestionRequest(BaseModel):
    records: list[IngestionRecordIn]


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    total: int
    succeeded: int
    failed: int
    errors: list[str] = []


class IndexStatsResponse(BaseModel):
    historical_chunk_count: int
    historical_card_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/historical/batch",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch ingest historical idea-card PPTs",
)
def ingest_historical_batch(
    request: BatchIngestionRequest,
    pipeline=Depends(get_ingestion_pipeline),
):
    """
    Ingest a batch of historical idea-card PPTs with known Value Stream mappings.

    Each record must include:
    - **file_path**: Absolute or relative path to the PPTX on the server.
    - **mapped_value_stream_ids**: Ground-truth VS IDs from JIRA/JITS.
    """
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No records provided.",
        )

    from src.ingestion.pipeline import IngestionRecord

    job_id = str(uuid4())
    records = [
        IngestionRecord(
            file_path=r.file_path,
            mapped_value_stream_ids=r.mapped_value_stream_ids,
            mapped_value_stream_names=r.mapped_value_stream_names,
            jira_ticket=r.jira_ticket,
            title=r.title,
            domain=r.domain,
        )
        for r in request.records
    ]

    results = pipeline.ingest_batch(records)
    succeeded = sum(1 for r in results if r.success)
    errors = [f"{r.file_path}: {r.error}" for r in results if not r.success]

    return IngestionJobResponse(
        job_id=job_id,
        status="completed",
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        errors=errors,
    )


@router.get(
    "/stats",
    response_model=IndexStatsResponse,
    summary="Return index statistics",
)
def get_index_stats(pipeline=Depends(get_ingestion_pipeline)):
    """Return counts of indexed chunks and idea cards."""
    # ChromaDB count
    try:
        chunk_count = pipeline._vector_store.count()
    except Exception:
        chunk_count = -1

    return IndexStatsResponse(
        historical_chunk_count=chunk_count,
        historical_card_count=-1,  # In production: query SQLite
    )
