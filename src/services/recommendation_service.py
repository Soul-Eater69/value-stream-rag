"""
RecommendationService: high-level facade for the recommendation workflow.

This class is the single entry point called by the API layer.
It handles upload management, workflow execution, and result retrieval.
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.graph.state import RecommendationState
from src.models.domain import RecommendationResult

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Orchestrates the full recommendation pipeline for an uploaded PPT.
    """

    def __init__(
        self,
        recommendation_graph,   # Compiled LangGraph
        upload_dir: str | Path = "./data/uploads",
    ) -> None:
        self._graph = recommendation_graph
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def recommend_from_file(
        self,
        file_path: str | Path,
        original_filename: str | None = None,
        domain_hint: str | None = None,
    ) -> RecommendationResult:
        """
        Run the full recommendation pipeline on a local PPTX file.

        This is the primary entry point for the API layer.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PPTX file not found: {file_path}")

        upload_id = str(uuid4())
        start_time = time.time()
        logger.info(
            "Starting recommendation | upload_id=%s filename=%s",
            upload_id,
            original_filename or path.name,
        )

        initial_state: RecommendationState = {
            "upload_id": upload_id,
            "file_path": str(path),
            "original_filename": original_filename or path.name,
            "domain_hint": domain_hint,
            "status": "starting",
            "errors": [],
            "retry_count": 0,
            "_start_time": start_time,
        }

        final_state = self._graph.invoke(initial_state)

        if final_state.get("status") == "error":
            errors = final_state.get("errors", [])
            raise RuntimeError(
                f"Recommendation workflow failed: {'; '.join(errors)}"
            )

        result = final_state.get("recommendation")
        if result is None:
            raise RuntimeError("Workflow completed but no recommendation was produced.")

        logger.info(
            "Recommendation complete | upload_id=%s | top_vs=%s | confidence=%.3f | ms=%d",
            upload_id,
            [vs.name for vs in result.recommended_value_streams[:3]],
            result.confidence,
            result.processing_duration_ms,
        )
        return result

    def save_upload(self, file_bytes: bytes, original_filename: str) -> Path:
        """Save raw upload bytes to the upload directory and return the path."""
        upload_id = str(uuid4())
        safe_name = Path(original_filename).stem[:80] + ".pptx"
        dest = self._upload_dir / upload_id / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file_bytes)
        logger.info("Saved upload: %s (%d bytes)", dest, len(file_bytes))
        return dest
