"""
Recommendation endpoints.

POST /api/v1/recommendations/upload
  Upload a PPTX and get Value Stream recommendations.

GET /api/v1/recommendations/{recommendation_id}
  Retrieve a previous recommendation by ID.

GET /api/v1/recommendations/{recommendation_id}/evidence
  Retrieve the supporting evidence for a recommendation.

POST /api/v1/recommendations/{recommendation_id}/feedback
  Submit human-in-the-loop feedback for evaluation.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.api.dependencies import get_recommendation_service
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────


class ValueStreamOut(BaseModel):
    id: str
    name: str
    description: str
    domain: str
    final_score: float | None
    stage_names: list[str] = []


class EvidenceOut(BaseModel):
    chunk_id: str
    chunk_content: str
    chunk_type: str
    slide_title: str
    source_document_path: str
    evidence_source: str
    similarity_score: float


class RecommendationResponse(BaseModel):
    recommendation_id: str
    upload_id: str
    query_summary: str
    recommended_value_streams: list[ValueStreamOut]
    reasoning: str
    confidence: float
    processing_duration_ms: int


class FeedbackRequest(BaseModel):
    approved: bool
    corrected_value_stream_ids: list[str] = []
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload idea-card PPT and get Value Stream recommendations",
)
async def upload_and_recommend(
    file: Annotated[UploadFile, File(description="Idea-card PowerPoint (.pptx)")],
    domain_hint: Annotated[str | None, Form()] = None,
    recommendation_svc=Depends(get_recommendation_service),
):
    """
    Upload a PPTX file and receive ranked Value Stream recommendations.

    - **file**: The idea-card PowerPoint file (`.pptx` required).
    - **domain_hint**: Optional business domain to narrow the search.

    Returns a ranked list of Value Streams with supporting evidence and reasoning.
    """
    if not file.filename or not file.filename.lower().endswith(".pptx"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .pptx files are supported.",
        )

    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    try:
        # Save to disk (recommendations are processed synchronously at PoC scale)
        saved_path = recommendation_svc.save_upload(file_bytes, file.filename)

        result = recommendation_svc.recommend_from_file(
            file_path=saved_path,
            original_filename=file.filename,
            domain_hint=domain_hint,
        )
    except RuntimeError as exc:
        logger.error("Recommendation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return RecommendationResponse(
        recommendation_id=str(result.recommendation_id),
        upload_id=result.upload_id,
        query_summary=result.query_summary,
        recommended_value_streams=[
            ValueStreamOut(
                id=vs.id,
                name=vs.name,
                description=vs.description,
                domain=vs.domain,
                final_score=vs.final_score,
                stage_names=vs.stage_sequence.stage_names if vs.stage_sequence else [],
            )
            for vs in result.recommended_value_streams
        ],
        reasoning=result.reasoning,
        confidence=result.confidence,
        processing_duration_ms=result.processing_duration_ms,
    )


@router.get(
    "/{recommendation_id}/evidence",
    summary="Retrieve supporting evidence for a recommendation",
)
async def get_evidence(recommendation_id: str):
    """
    Returns the evidence chunks that supported a given recommendation.

    Evidence is stored in the recommendation trace and can be used to
    explain why each Value Stream was recommended.
    """
    # In production: fetch from DB via recommendation ID
    # PoC: return placeholder
    return {
        "recommendation_id": recommendation_id,
        "evidence": [],
        "note": "Evidence retrieval requires DB integration (see RecommendationTraceORM).",
    }


@router.post(
    "/{recommendation_id}/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit human feedback for a recommendation",
)
async def submit_feedback(recommendation_id: str, feedback: FeedbackRequest):
    """
    Record human-in-the-loop feedback for evaluation and model improvement.

    - **approved**: Whether the recommendation was accepted.
    - **corrected_value_stream_ids**: Correct VS IDs if recommendation was wrong.
    """
    logger.info(
        "Feedback received | rec_id=%s approved=%s corrected=%s",
        recommendation_id,
        feedback.approved,
        feedback.corrected_value_stream_ids,
    )
    # In production: update RecommendationTraceORM.human_approved etc.
    return {"status": "accepted", "recommendation_id": recommendation_id}
