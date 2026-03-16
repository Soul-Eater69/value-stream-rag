"""
LangGraph nodes for the PPT parsing, chunking, and pre-retrieval gating stage.

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
from src.models.domain import ChunkType, UploadedPPT

logger = logging.getLogger(__name__)

# ── Long-document gate ────────────────────────────────────────────────────────
# Above this slide count the mean-pool of all chunk embeddings becomes noisy
# (unrelated content dilutes the signal).  The gate trims query_chunks to
# title and synthetic-summary chunks only, and signals the retriever to skip
# per-chunk ChromaDB search in favour of doc-summary-only retrieval.
LONG_DOC_SLIDE_THRESHOLD = 40


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
            "_parsed_document": parsed,
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


def apply_long_doc_gate(state: RecommendationState) -> RecommendationState:
    """
    Node: Detect long documents and trim query chunks to title/summary only.

    For decks above LONG_DOC_SLIDE_THRESHOLD slides, mean-pooling all chunk
    embeddings dilutes the query signal with noise from unrelated content.
    This node:
      1. Detects whether the deck is long (via parsed slide count or chunk count).
      2. If long: retains only SLIDE_TITLE and SYNTHETIC_SUMMARY chunks.
         Falls back to the first N chunks if no title/summary chunks exist.
      3. Sets state["long_doc_mode"] = True so the retriever can skip
         per-chunk ChromaDB search and use doc-summary-only retrieval.

    Input  state keys: query_chunks, _parsed_document
    Output state keys: query_chunks (possibly trimmed), long_doc_mode
    """
    parsed = state.get("_parsed_document")
    chunks = state.get("query_chunks", [])

    slide_count = getattr(parsed, "slide_count", len(chunks))

    if slide_count <= LONG_DOC_SLIDE_THRESHOLD:
        return {**state, "long_doc_mode": False}

    logger.info(
        "Long-doc gate triggered: slide_count=%d > threshold=%d | upload_id=%s",
        slide_count,
        LONG_DOC_SLIDE_THRESHOLD,
        state.get("upload_id"),
    )

    # Prefer title + synthetic summary chunks for a clean query vector
    summary_types = {ChunkType.SLIDE_TITLE, ChunkType.SYNTHETIC_SUMMARY}
    filtered = [c for c in chunks if c.chunk_type in summary_types]

    if not filtered:
        # Fallback: one chunk per slide (every slide_index change), up to threshold
        seen_slides: set[int] = set()
        filtered = []
        for c in chunks:
            if c.slide_index not in seen_slides:
                filtered.append(c)
                seen_slides.add(c.slide_index)
            if len(filtered) >= LONG_DOC_SLIDE_THRESHOLD:
                break

    logger.info(
        "Long-doc gate: %d chunks → %d title/summary chunks",
        len(chunks),
        len(filtered),
    )
    return {**state, "query_chunks": filtered, "long_doc_mode": True}


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
