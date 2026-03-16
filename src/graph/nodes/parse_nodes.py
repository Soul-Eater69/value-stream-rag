"""
LangGraph nodes for the PPT parsing and chunking stage.

Nodes are pure functions (state → state) so they can be composed,
tested, and retried independently.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.chunking.chunker import ChunkingConfig, PPTChunker
from src.graph.state import RecommendationState
from src.ingestion.parser import PPTParser
from src.models.domain import UploadedPPT

logger = logging.getLogger(__name__)


def parse_uploaded_ppt(state: RecommendationState) -> RecommendationState:
    """
    Node: Parse the uploaded PPTX file using MarkItDown via PPTParser.

    Input  state keys: upload_id, file_path, original_filename
    Output state keys: uploaded_ppt, status
    """
    logger.info("Node: parse_uploaded_ppt | upload_id=%s", state.get("upload_id"))

    try:
        parser = PPTParser()
        parsed = parser.parse(state["file_path"])

        path = Path(state["file_path"])
        ppt = UploadedPPT(
            upload_id=state["upload_id"],
            original_filename=state.get("original_filename", path.name),
            file_path=state["file_path"],
            file_size_bytes=path.stat().st_size,
            processing_status="parsed",
        )

        return {
            **state,
            "uploaded_ppt": ppt,
            "_parsed_document": parsed,  # Temporary; consumed by next node
            "status": "parsed",
            "errors": state.get("errors", []),
        }

    except Exception as exc:
        logger.exception("parse_uploaded_ppt failed")
        errors = state.get("errors", [])
        errors.append(f"ParseError: {exc}")
        return {**state, "status": "error", "errors": errors}


def chunk_uploaded_ppt(state: RecommendationState) -> RecommendationState:
    """
    Node: Chunk the parsed PPT document.

    Input  state keys: _parsed_document, upload_id
    Output state keys: query_chunks
    """
    logger.info("Node: chunk_uploaded_ppt | upload_id=%s", state.get("upload_id"))

    try:
        parsed = state.get("_parsed_document")
        if parsed is None:
            raise ValueError("No parsed document in state; parse node must run first.")

        chunker = PPTChunker(ChunkingConfig())
        chunks = chunker.chunk_document(parsed, source_document_id=state["upload_id"])

        logger.info("Produced %d chunks for upload_id=%s", len(chunks), state["upload_id"])
        return {**state, "query_chunks": chunks, "status": "chunked"}

    except Exception as exc:
        logger.exception("chunk_uploaded_ppt failed")
        errors = state.get("errors", [])
        errors.append(f"ChunkError: {exc}")
        return {**state, "status": "error", "errors": errors}


def embed_query_chunks(state: RecommendationState, embedding_service) -> RecommendationState:
    """
    Node (closure): Embed all query chunks.

    Input  state keys: query_chunks
    Output state keys: chunk_embeddings
    """
    logger.info("Node: embed_query_chunks | upload_id=%s", state.get("upload_id"))

    try:
        chunks = state.get("query_chunks", [])
        if not chunks:
            raise ValueError("No chunks to embed; chunk node must run first.")

        texts = [c.embedding_text for c in chunks]
        embeddings = embedding_service.embed_batch(texts)

        return {**state, "chunk_embeddings": embeddings, "status": "embedded"}

    except Exception as exc:
        logger.exception("embed_query_chunks failed")
        errors = state.get("errors", [])
        errors.append(f"EmbedError: {exc}")
        return {**state, "status": "error", "errors": errors}
