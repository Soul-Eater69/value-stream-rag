"""
Unit tests for evaluation metrics.
"""

from __future__ import annotations

import math
import pytest

from src.evaluation.evaluator import CardEvaluation, EvaluationReport


class TestCardEvaluation:
    def _make_eval(self, gold: list[str], predicted: list[str]) -> CardEvaluation:
        e = CardEvaluation(
            card_id="c1",
            file_path="/fake.pptx",
            gold_vs_ids=gold,
            predicted_vs_ids=predicted,
        )
        e.compute()
        return e

    def test_perfect_hit_at_1(self):
        e = self._make_eval(["VS001"], ["VS001", "VS002", "VS003"])
        assert e.hit_at_1 is True
        assert e.mrr == pytest.approx(1.0)

    def test_miss_at_1_hit_at_3(self):
        e = self._make_eval(["VS003"], ["VS001", "VS002", "VS003", "VS004"])
        assert e.hit_at_1 is False
        assert e.hit_at_3 is True
        assert e.mrr == pytest.approx(1.0 / 3)

    def test_precision_at_5_partial(self):
        e = self._make_eval(["VS001", "VS002"], ["VS001", "VS003", "VS004", "VS005", "VS002"])
        # 2 hits in top 5
        assert e.precision_at_5 == pytest.approx(2 / 5)

    def test_recall_at_5(self):
        e = self._make_eval(["VS001", "VS002", "VS006"], ["VS001", "VS002", "VS003"])
        # 2/3 gold VSes found
        assert e.recall_at_5 == pytest.approx(2 / 3)

    def test_ndcg_perfect(self):
        e = self._make_eval(["VS001"], ["VS001"])
        assert e.ndcg_at_5 == pytest.approx(1.0)

    def test_ndcg_zero(self):
        e = self._make_eval(["VS001"], ["VS002", "VS003"])
        assert e.ndcg_at_5 == pytest.approx(0.0)

    def test_ndcg_partial(self):
        e = self._make_eval(["VS001", "VS002"], ["VS003", "VS001", "VS002"])
        # Not a perfect ranking – VS001 at pos 2, VS002 at pos 3
        assert 0 < e.ndcg_at_5 < 1.0

    def test_empty_gold(self):
        e = self._make_eval([], ["VS001"])
        assert e.precision_at_5 == 0.0
        assert e.recall_at_5 == 0.0

    def test_empty_predictions(self):
        e = self._make_eval(["VS001"], [])
        assert e.hit_at_1 is False
        assert e.mrr == 0.0


class TestEvaluationReport:
    def test_aggregate_computation(self):
        results = []
        for i in range(3):
            e = CardEvaluation(
                card_id=f"c{i}",
                file_path=f"/fake{i}.pptx",
                gold_vs_ids=["VS001"],
                predicted_vs_ids=["VS001"],
            )
            e.compute()
            results.append(e)

        report = EvaluationReport(total_cards=3, per_card_results=results)
        report.compute_aggregates()

        assert report.avg_hit_at_1 == pytest.approx(1.0)
        assert report.avg_mrr == pytest.approx(1.0)
        assert report.evaluated_cards == 3

    def test_failed_cards_excluded(self):
        results = []
        e1 = CardEvaluation("c1", "/f1.pptx", ["VS001"], ["VS001"])
        e1.compute()
        results.append(e1)

        e2 = CardEvaluation("c2", "/f2.pptx", ["VS001"], [], error="FileNotFound")
        results.append(e2)

        report = EvaluationReport(total_cards=2, failed_cards=1, per_card_results=results)
        report.compute_aggregates()
        assert report.evaluated_cards == 1  # Only valid card counted
