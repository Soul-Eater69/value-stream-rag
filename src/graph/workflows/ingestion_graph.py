"""
LangGraph ingestion workflow for batch processing of historical PPTs.

Graph topology:
  START
    │
    ▼
  validate_records
    │
    ▼
  process_file (iterates per file)
    │
    ├── parse_file
    │      │
    │      ▼
    │   chunk_file
    │      │
    │      ▼
    │   embed_file
    │      │
    │      ▼
    │   index_file ──► update_results
    │
    ▼
  finalize_job
    │
    ▼
   END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.graph.state import IngestionState

logger = logging.getLogger(__name__)


def build_ingestion_graph(ingestion_pipeline) -> StateGraph:
    """Build and return the compiled ingestion workflow."""

    def validate_records(state: IngestionState) -> IngestionState:
        records = state.get("input_records", [])
        valid = [r for r in records if r.get("file_path") and r.get("mapped_value_stream_ids")]
        invalid_count = len(records) - len(valid)
        if invalid_count > 0:
            logger.warning("Skipping %d invalid ingestion records", invalid_count)

        errors = state.get("errors", [])
        if invalid_count:
            errors.append(f"Skipped {invalid_count} records missing file_path or mapped_vs_ids")

        return {
            **state,
            "input_records": valid,
            "results": [],
            "errors": errors,
            "status": "validated",
        }

    def process_all_files(state: IngestionState) -> IngestionState:
        """Process all records through the IngestionPipeline."""
        from src.ingestion.pipeline import IngestionRecord

        records = state.get("input_records", [])
        results = []
        errors = state.get("errors", [])

        for raw in records:
            try:
                record = IngestionRecord(**raw)
                result = ingestion_pipeline.ingest_single(record)
                results.append(
                    {
                        "card_id": result.card_id,
                        "file_path": result.file_path,
                        "slide_count": result.slide_count,
                        "chunk_count": result.chunk_count,
                        "success": result.success,
                        "error": result.error,
                    }
                )
                if not result.success:
                    errors.append(f"{raw['file_path']}: {result.error}")
            except Exception as exc:
                logger.exception("Failed to process record: %s", raw.get("file_path"))
                errors.append(f"{raw.get('file_path', 'unknown')}: {exc}")

        return {
            **state,
            "results": results,
            "errors": errors,
            "status": "processed",
        }

    def finalize_job(state: IngestionState) -> IngestionState:
        results = state.get("results", [])
        success = sum(1 for r in results if r.get("success"))
        failed = len(results) - success
        logger.info("Ingestion job completed: %d success / %d failed", success, failed)
        return {**state, "status": "completed"}

    # Build graph
    graph = StateGraph(IngestionState)
    graph.add_node("validate", validate_records)
    graph.add_node("process", process_all_files)
    graph.add_node("finalize", finalize_job)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "process")
    graph.add_edge("process", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
