"""
ValueStreamRanker: aggregates multi-source evidence into a final ranked list.

Scoring formula (configurable weights):
  final_score = (
      w_vs   * vs_similarity_score      # Azure AI Search score
    + w_hist * historical_boost_score   # Boost from historical matches
    + w_rer  * rerank_score             # Azure semantic / cross-encoder score
    + w_stg  * stage_match_score        # Stage keyword overlap with PPT content
  )

Historical boost: proportional to the number of historically similar idea cards
that were mapped to this VS, weighted by their similarity to the uploaded PPT.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from src.models.domain import EvidenceSource, RetrievalEvidence, ValueStream
from src.search.retriever import RetrievalContext

logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    weight_vs_similarity: float = 0.40
    weight_historical: float = 0.35
    weight_rerank: float = 0.15
    weight_stage_match: float = 0.10
    top_n: int = 5


class ValueStreamRanker:
    """
    Ranks Value Stream candidates by combining evidence from multiple sources.
    """

    def __init__(self, config: RankingConfig | None = None) -> None:
        self.cfg = config or RankingConfig()

    def rank(
        self,
        ctx: RetrievalContext,
        query_text: str,
    ) -> list[ValueStream]:
        """
        Produce a ranked list of Value Streams.

        1. Start with Azure AI Search candidates.
        2. Include any additional VSes surfaced only by historical evidence.
        3. Score each VS using the weighted formula.
        4. Return top-N.
        """
        # Collect all unique VS IDs seen across both sources
        all_vs_ids: set[str] = {vs.id for vs in ctx.vs_candidates}
        all_vs_ids |= ctx.historically_matched_vs_ids

        # Build ID → ValueStream map from Azure candidates
        vs_map: dict[str, ValueStream] = {vs.id: vs for vs in ctx.vs_candidates}

        # Historical boost: map VS ID → aggregated boost score
        hist_boost = self._compute_historical_boost(ctx)

        # Stage match: map VS ID → stage overlap score
        stage_scores = self._compute_stage_scores(ctx, query_text)

        scored: list[tuple[float, ValueStream]] = []
        for vs_id in all_vs_ids:
            vs = vs_map.get(vs_id)
            if vs is None:
                # VS was only seen in historical evidence; skip if no VS metadata
                continue

            vs_sim = vs.similarity_score or 0.0
            rer = vs.rerank_score or vs_sim  # Fall back to sim if no rerank score
            hist = hist_boost.get(vs_id, 0.0)
            stage = stage_scores.get(vs_id, 0.0)

            final = (
                self.cfg.weight_vs_similarity * vs_sim
                + self.cfg.weight_historical * hist
                + self.cfg.weight_rerank * rer
                + self.cfg.weight_stage_match * stage
            )
            vs.final_score = round(final, 4)
            scored.append((final, vs))

        # Sort descending by final score
        scored.sort(key=lambda t: t[0], reverse=True)

        top_n = [vs for _, vs in scored[: self.cfg.top_n]]
        logger.info(
            "Ranked %d VS candidates → top %d: %s",
            len(scored),
            self.cfg.top_n,
            [vs.name for vs in top_n],
        )
        return top_n

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _compute_historical_boost(
        self, ctx: RetrievalContext
    ) -> dict[str, float]:
        """
        Aggregate historical evidence into per-VS boost scores.

        Boost = mean similarity of chunks that reference this VS,
                normalised to [0, 1].
        """
        vs_sim_sums: dict[str, float] = defaultdict(float)
        vs_hit_counts: dict[str, int] = defaultdict(int)

        for hit in ctx.historical_hits:
            for vs_id in hit.get("mapped_vs_ids", []):
                vs_sim_sums[vs_id] += hit.get("similarity", 0.0)
                vs_hit_counts[vs_id] += 1

        boost: dict[str, float] = {}
        for vs_id, total_sim in vs_sim_sums.items():
            count = vs_hit_counts[vs_id]
            # Average similarity, boosted by log(count) to reward consensus
            import math
            boost[vs_id] = min(1.0, (total_sim / count) * (1 + 0.1 * math.log1p(count)))

        return boost

    def _compute_stage_scores(
        self, ctx: RetrievalContext, query_text: str
    ) -> dict[str, float]:
        """
        Compute keyword-overlap score between VS stage keywords and query text.
        """
        query_lower = query_text.lower()
        scores: dict[str, float] = {}

        for vs in ctx.vs_candidates:
            if vs.stage_sequence is None:
                scores[vs.id] = 0.0
                continue

            all_stage_keywords: list[str] = []
            for stage in vs.stage_sequence.stages:
                all_stage_keywords.extend(stage.keywords)
                all_stage_keywords.append(stage.name.lower())

            if not all_stage_keywords:
                scores[vs.id] = 0.0
                continue

            matched = sum(1 for kw in all_stage_keywords if kw.lower() in query_lower)
            scores[vs.id] = min(1.0, matched / len(all_stage_keywords))

        return scores
