"""
FastAPI dependency injection helpers.

The ServiceContainer singleton is created once at startup and injected
into route handlers via FastAPI's Depends() mechanism.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from src.services.container import ServiceContainer


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    """Return the application-wide service container (singleton)."""
    return ServiceContainer()


def get_recommendation_service(
    container: ServiceContainer = Depends(get_container),
):
    from src.services.recommendation_service import RecommendationService
    from src.config.settings import get_settings
    settings = get_settings()
    return RecommendationService(
        recommendation_graph=container.recommendation_graph,
        upload_dir=settings.upload_dir,
    )


def get_ingestion_pipeline(
    container: ServiceContainer = Depends(get_container),
):
    return container.ingestion_pipeline
