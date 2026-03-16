"""
ExhaustiveBaselineRetriever: pure cosine scoring against all Value Streams.

Purpose
-------
With ~50 Value Streams, ranking them purely by cosine similarity of the
upload's mean-pooled embedding against each VS description embedding is
both the simplest possible algorithm AND a strong baseline.

Use this in two ways:

  1. **Benchmark** — run alongside the full HybridRetriever in offline eval
     to measure whether historical boost + stage scoring actually improve
     Hit@K / MRR over the cosine-only baseline.  If cosine-only matches or
     beats the full pipeline, you can retire the complexity.

  2. **Fallback** — if ChromaDB is unavailable (first-run, cold start) the
     baseline still returns useful results.

Integration
-----------
Drop ExhaustiveBaselineRetriever in wherever HybridRetriever is expected;
it satisfies the same interface (retrieve → RetrievalContext).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.models.domain import Chunk, ChunkType, EvidenceSource, RetrievalEvidence
from src.search.retriever import RetrievalContext

logger = logging.getLogger(__name__)


class ExhaustiveBaselineRetriever:
    """
    Retriever that scores all VSes via cosine similarity only.

    No ChromaDB.  No historical boost.  No LLM.
    The RetrievalContext it returns has:
      - vs_candidates: all VSes with similarity_score set
      - historical_hits: empty
      - historically_matched_vs_ids: empty
      - evidence: one VALUE_STREAM_INDEX entry per VS (cosine score)
    """

    def __init__(self, vs_catalogue, embedding_service) -> None:
        self._catalogue = vs_catalogue
        self._embedder = embedding_service

    def retrieve(
        self,
        query_chunks: list[Chunk],
        domain_filter: str | None = None,
    ) -> RetrievalContext:
        ctx = RetrievalContext()

        if not query_chunks:
            return ctx

        texts = [c.embedding_text for c in query_chunks]
        chunk_embeddings = self._embedder.embed_batch(texts)
        agg_embedding = _mean_pool(chunk_embeddings)

        cosine_scores = self._catalogue.score_all(agg_embedding)

        vs_candidates = []
        for vs_id, score in cosine_scores.items():
            vs = self._catalogue.all()[vs_id]
            if domain_filter and vs.domain and vs.domain != domain_filter:
                continue
            scored_vs = vs.model_copy()
            scored_vs.similarity_score = round(score, 4)
            vs_candidates.append(scored_vs)
            ctx.evidence[vs_id].append(
                RetrievalEvidence(
                    chunk_id=f"baseline_vs_{vs_id}",
                    chunk_content=vs.description,
                    chunk_type=ChunkType.SLIDE_TEXT,
                    slide_index=-1,
                    slide_title="Value Stream Definition (baseline)",
                    source_document_id=vs_id,
                    source_document_path="vs_catalogue_baseline",
                    evidence_source=EvidenceSource.VALUE_STREAM_INDEX,
                    similarity_score=score,
                )
            )

        ctx.vs_candidates = vs_candidates
        logger.debug("Baseline retrieved %d VS candidates", len(vs_candidates))
        return ctx


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
