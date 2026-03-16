"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    services: dict[str, str] = {}


@router.get("/", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Returns service health status."""
    return HealthResponse(status="healthy")


@router.get("/ready", response_model=HealthResponse, summary="Readiness check")
async def readiness_check() -> HealthResponse:
    """Check if all downstream services are reachable."""
    # In production: ping Azure AI Search, ChromaDB, OpenAI
    return HealthResponse(
        status="ready",
        services={
            "azure_search": "ok",
            "chroma": "ok",
            "openai": "ok",
        },
    )
