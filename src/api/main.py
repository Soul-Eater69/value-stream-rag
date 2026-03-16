"""
FastAPI application entry point.

Registers all routers, configures middleware, and sets up the service container
via dependency injection.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.logging_config import configure_logging
from src.config.settings import get_settings

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Value Stream RAG API",
    description=(
        "Enterprise RAG system that recommends relevant Value Streams "
        "for uploaded idea-card PowerPoint presentations."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


# ── Routers
from src.api.routers import health, recommendations, ingestion  # noqa: E402

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import structlog
    logger = structlog.get_logger()
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
