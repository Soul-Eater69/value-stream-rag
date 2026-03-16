"""
Unit tests for the ValueStreamRanker.
"""

from __future__ import annotations

import pytest

from src.models.domain import Stage, StageSequence, ValueStream
from src.ranking.ranker import RankingConfig, ValueStreamRanker
from src.search.retriever import RetrievalContext


def _make_vs(
    vs_id: str,
    name: str,
    similarity: float = 0.7,
    rerank: float | None = None,
    stage_keywords: list[str] | None = None,
) -> ValueStream:
    stages = []
    if stage_keywords:
        stages = [
            Stage(id=f"{vs_id}-S1", name="Stage1", sequence_order=1, keywords=stage_keywords)
        ]
    stage_seq = StageSequence(value_stream_id=vs_id, stages=stages) if stages else None

    return ValueStream(
        id=vs_id,
        name=name,
        description=f"Description of {name}",
        domain="IT",
        similarity_score=similarity,
        rerank_score=rerank,
        stage_sequence=stage_seq,
    )


@pytest.fixture
def basic_context() -> RetrievalContext:
    ctx = RetrievalContext()
    ctx.vs_candidates = [
        _make_vs("VS001", "Order Management", similarity=0.85),
        _make_vs("VS002", "Inventory Control", similarity=0.72),
        _make_vs("VS003", "Customer Onboarding", similarity=0.60),
    ]
    return ctx


class TestValueStreamRanker:
    def test_returns_top_n(self, basic_context):
        cfg = RankingConfig(top_n=2)
        ranker = ValueStreamRanker(cfg)
        ranked = ranker.rank(basic_context, query_text="order fulfilment problem")
        assert len(ranked) <= 2

    def test_higher_similarity_ranks_first(self, basic_context):
        ranker = ValueStreamRanker(RankingConfig(top_n=3))
        ranked = ranker.rank(basic_context, query_text="order")
        # Without historical boost, VS001 (sim=0.85) should rank first
        assert ranked[0].id == "VS001"

    def test_final_score_set(self, basic_context):
        ranker = ValueStreamRanker()
        ranked = ranker.rank(basic_context, query_text="order")
        assert all(vs.final_score is not None for vs in ranked)

    def test_historical_boost_applied(self):
        ctx = RetrievalContext()
        ctx.vs_candidates = [
            _make_vs("VS001", "Alpha", similarity=0.50),
            _make_vs("VS002", "Beta", similarity=0.60),
        ]
        # Simulate strong historical evidence for VS001
        ctx.historical_hits = [
            {
                "id": "chunk-1",
                "document": "historical text",
                "similarity": 0.95,
                "source_document_id": "card-001",
                "source_document_path": "/path/card001.pptx",
                "slide_index": 0,
                "slide_title": "Problem",
                "chunk_type": "slide_text",
                "section_label": "",
                "table_index": -1,
                "mapped_vs_ids": ["VS001"],
            }
        ]
        ctx.historically_matched_vs_ids = {"VS001"}

        ranker = ValueStreamRanker(RankingConfig(top_n=2, weight_historical=0.5, weight_vs_similarity=0.3))
        ranked = ranker.rank(ctx, query_text="problem statement")
        # VS001 should now outrank VS002 despite lower similarity
        assert ranked[0].id == "VS001"

    def test_stage_keyword_boost(self):
        ctx = RetrievalContext()
        ctx.vs_candidates = [
            _make_vs("VS001", "Alpha", similarity=0.70, stage_keywords=["inventory", "stock", "order"]),
            _make_vs("VS002", "Beta", similarity=0.70),
        ]

        ranker = ValueStreamRanker(RankingConfig(top_n=2, weight_stage_match=0.4, weight_vs_similarity=0.4))
        ranked = ranker.rank(ctx, query_text="we have an inventory and order management problem")
        # VS001 should win due to stage keyword overlap
        assert ranked[0].id == "VS001"

    def test_empty_candidates(self):
        ctx = RetrievalContext()
        ranker = ValueStreamRanker()
        ranked = ranker.rank(ctx, query_text="something")
        assert ranked == []
