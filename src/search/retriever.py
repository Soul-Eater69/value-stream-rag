"""
HybridRetriever: orchestrates multi-source retrieval for the recommendation pipeline.

Retrieval strategy:
  1. Embed uploaded PPT chunks and mean-pool into a single query vector.
  2. Score ALL Value Streams via in-memory cosine similarity (VSCatalogue).
  3. Retrieve similar historical idea-card chunks from ChromaDB.
  4. Extract VS IDs from historical hits (mapped_vs_ids in metadata).
  5. Return combined RetrievalContext for the ranking module.

Design note:
  With ~50 Value Streams, scoring all of them in memory is faster and more
  complete than a top-K index search.  No VS can be silently dropped because
  Azure didn't happen to return it — every VS gets a cosine score regardless
  of whether it also appears in historical evidence.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.models.domain import (
    Chunk,
    ChunkType,
    EvidenceSource,
    RetrievalEvidence,
    ValueStream,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Carries all retrieval outputs to the ranking layer."""

    # All Value Streams scored against the upload (full catalogue, scored)
    vs_candidates: list[ValueStream] = field(default_factory=list)

    # Raw hits from ChromaDB (historical chunks)
    historical_hits: list[dict[str, Any]] = field(default_factory=list)

    # Evidence map: VS ID → list of supporting evidence
    evidence: dict[str, list[RetrievalEvidence]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # VS IDs referenced by historically similar idea cards
    historically_matched_vs_ids: set[str] = field(default_factory=set)


class HybridRetriever:
    """
    Combines VSCatalogue (all Value Streams, in-memory cosine) + ChromaDB
    (historical idea-card chunks) to produce a rich RetrievalContext.
    """

    def __init__(
        self,
        vs_catalogue,           # VSCatalogue — pre-loaded at startup
        chroma_store,           # ChromaVectorStore
        embedding_service,      # EmbeddingService
        top_k_hist: int = 20,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._catalogue = vs_catalogue
        self._chroma = chroma_store
        self._embedder = embedding_service
        self._top_k_hist = top_k_hist
        self._threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_chunks: list[Chunk],
        domain_filter: str | None = None,
        skip_historical: bool = False,
    ) -> RetrievalContext:
        """
        Run multi-source retrieval for a list of query chunks (from uploaded PPT).

        Strategy:
          - Embed each chunk individually, then aggregate via mean-pooling.
          - Score ALL VSes via cosine against the aggregated embedding.
          - Query ChromaDB per-chunk for historical matches.
        """
        ctx = RetrievalContext()

        if not query_chunks:
            logger.warning("No query chunks provided to retriever")
            return ctx

        # Embed all chunks
        texts = [c.embedding_text for c in query_chunks]
        chunk_embeddings = self._embedder.embed_batch(texts)

        # Aggregate query: mean-pool all chunk embeddings
        agg_embedding = self._mean_pool(chunk_embeddings)

        # ── Step 1: Score all Value Streams via in-memory cosine
        logger.info(
            "Scoring all %d Value Streams via in-memory cosine", self._catalogue.size
        )
        cosine_scores = self._catalogue.score_all(agg_embedding)

        vs_candidates: list[ValueStream] = []
        for vs_id, score in cosine_scores.items():
            vs = self._catalogue.all()[vs_id]
            if domain_filter and vs.domain and vs.domain != domain_filter:
                continue
            # Use model_copy to avoid mutating the shared catalogue object
            scored_vs = vs.model_copy()
            scored_vs.similarity_score = round(score, 4)
            vs_candidates.append(scored_vs)
            ctx.evidence[vs_id].append(
                RetrievalEvidence(
                    chunk_id=f"vs_{vs_id}",
                    chunk_content=vs.description,
                    chunk_type=ChunkType.SLIDE_TEXT,
                    slide_index=-1,
                    slide_title="Value Stream Definition",
                    source_document_id=vs_id,
                    source_document_path="vs_catalogue",
                    evidence_source=EvidenceSource.VALUE_STREAM_INDEX,
                    similarity_score=score,
                )
            )

        ctx.vs_candidates = vs_candidates
        logger.info("Catalogue scoring produced %d VS candidates", len(vs_candidates))

        # ── Step 2: Historical PPT retrieval from ChromaDB
        # Skipped in long-doc mode: the trimmed title-only chunks are too sparse
        # to produce useful per-chunk similarity matches.
        if skip_historical:
            logger.info(
                "skip_historical=True (long-doc mode); skipping ChromaDB per-chunk search"
            )
            return ctx

        logger.info("Running vector search against historical PPT index (ChromaDB)")
        all_hist_hits: list[dict] = []
        per_chunk_k = max(1, self._top_k_hist // len(query_chunks))
        for chunk, embedding in zip(query_chunks, chunk_embeddings):
            hits = self._chroma.query(
                query_embedding=embedding,
                top_k=per_chunk_k,
            )
            hits = [h for h in hits if h["similarity"] >= self._threshold]
            for hit in hits:
                hit["_query_chunk_id"] = str(chunk.id)
            all_hist_hits.extend(hits)

        ctx.historical_hits = all_hist_hits
        logger.info(
            "ChromaDB returned %d historical hits (above threshold)", len(all_hist_hits)
        )

        # ── Step 3: Build evidence from historical hits
        for hit in all_hist_hits:
            for vs_id in hit.get("mapped_vs_ids", []):
                ctx.historically_matched_vs_ids.add(vs_id)
                ctx.evidence[vs_id].append(
                    RetrievalEvidence(
                        chunk_id=hit["id"],
                        chunk_content=hit["document"],
                        chunk_type=hit.get("chunk_type", ChunkType.SLIDE_TEXT),
                        slide_index=hit.get("slide_index", 0),
                        slide_title=hit.get("slide_title", ""),
                        source_document_id=hit.get("source_document_id", ""),
                        source_document_path=hit.get("source_document_path", ""),
                        historical_card_id=hit.get("source_document_id"),
                        historical_mapped_vs_ids=hit.get("mapped_vs_ids", []),
                        evidence_source=EvidenceSource.HISTORICAL_PPT_INDEX,
                        similarity_score=hit.get("similarity", 0.0),
                    )
                )

        return ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        dim = len(embeddings[0])
        pooled = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                pooled[i] += v
        n = len(embeddings)
        return [v / n for v in pooled]

    @staticmethod
    def _build_query_text(chunks: list[Chunk]) -> str:
        """Build a condensed text query from chunk titles and content."""
        parts: list[str] = []
        seen_titles: set[str] = set()
        for c in chunks:
            if c.slide_title and c.slide_title not in seen_titles:
                parts.append(c.slide_title)
                seen_titles.add(c.slide_title)
            if c.chunk_type == ChunkType.SLIDE_TEXT and len(c.content) < 300:
                parts.append(c.content)
        return " ".join(parts)[:2000]
