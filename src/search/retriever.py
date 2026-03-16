"""
HybridRetriever: orchestrates multi-source retrieval for the recommendation pipeline.

Retrieval strategy:
  1. Generate an aggregated query from uploaded PPT chunks.
  2. Retrieve candidate Value Streams from Azure AI Search (hybrid search).
  3. Retrieve similar historical idea-card chunks from ChromaDB.
  4. Extract and boost Value Streams referenced in historical matches.
  5. Return combined evidence for the ranking module.
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

    # Value Streams from Azure AI Search
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
    Combines Azure AI Search (Value Streams) + ChromaDB (historical PPTs)
    to produce a rich RetrievalContext.
    """

    def __init__(
        self,
        azure_searcher,     # AzureValueStreamSearcher
        chroma_store,       # ChromaVectorStore
        embedding_service,  # EmbeddingService
        top_k_vs: int = 10,
        top_k_hist: int = 20,
        hybrid_alpha: float = 0.5,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._azure = azure_searcher
        self._chroma = chroma_store
        self._embedder = embedding_service
        self._top_k_vs = top_k_vs
        self._top_k_hist = top_k_hist
        self._hybrid_alpha = hybrid_alpha
        self._threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_chunks: list[Chunk],
        domain_filter: str | None = None,
    ) -> RetrievalContext:
        """
        Run multi-source retrieval for a list of query chunks (from uploaded PPT).

        Strategy:
          - Embed each chunk individually, then aggregate via mean-pooling.
          - Use the aggregated embedding for Azure AI Search (Value Streams).
          - Use per-chunk embeddings for ChromaDB (historical matches).
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
        agg_text = self._build_query_text(query_chunks)

        # ── Step 1: Value Stream retrieval from Azure AI Search
        logger.info("Running hybrid search against Azure AI Search index")
        vs_candidates = self._azure.hybrid_search(
            query_text=agg_text,
            query_embedding=agg_embedding,
            top_k=self._top_k_vs,
            domain_filter=domain_filter,
            alpha=self._hybrid_alpha,
        )
        ctx.vs_candidates = vs_candidates
        logger.info("Azure AI Search returned %d VS candidates", len(vs_candidates))

        # Populate evidence from VS hits
        for vs in vs_candidates:
            ctx.evidence[vs.id].append(
                RetrievalEvidence(
                    chunk_id=f"vs_{vs.id}",
                    chunk_content=vs.description,
                    chunk_type=ChunkType.SLIDE_TEXT,
                    slide_index=-1,
                    slide_title="Value Stream Definition",
                    source_document_id=vs.id,
                    source_document_path="azure_ai_search",
                    evidence_source=EvidenceSource.VALUE_STREAM_INDEX,
                    similarity_score=vs.similarity_score or 0.0,
                    rerank_score=vs.rerank_score,
                )
            )

        # ── Step 2: Historical PPT retrieval from ChromaDB
        logger.info("Running vector search against historical PPT index (ChromaDB)")
        all_hist_hits: list[dict] = []
        for chunk, embedding in zip(query_chunks, chunk_embeddings):
            hits = self._chroma.query(
                query_embedding=embedding,
                top_k=self._top_k_hist // max(len(query_chunks), 1),
            )
            # Filter by threshold
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
        # Prioritise titles and short text chunks; limit total length
        parts: list[str] = []
        seen_titles: set[str] = set()
        for c in chunks:
            if c.slide_title and c.slide_title not in seen_titles:
                parts.append(c.slide_title)
                seen_titles.add(c.slide_title)
            if c.chunk_type == ChunkType.SLIDE_TEXT and len(c.content) < 300:
                parts.append(c.content)
        return " ".join(parts)[:2000]  # Truncate to safe BM25 length
