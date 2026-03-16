"""
Offline evaluation framework for recommendation quality.

Metrics computed:
  - Hit@K        : Is any gold VS in the top-K recommendations?
  - Precision@K  : What fraction of top-K are gold VSes?
  - Recall@K     : What fraction of gold VSes are in top-K?
  - MRR          : Mean Reciprocal Rank of the first gold VS hit
  - NDCG@K       : Normalised Discounted Cumulative Gain

Each metric is computed per-card and then averaged across the evaluation set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.domain import RecommendationResult

logger = logging.getLogger(__name__)


@dataclass
class CardEvaluation:
    """Evaluation result for a single idea card."""

    card_id: str
    file_path: str
    gold_vs_ids: list[str]  # Ground-truth from JIRA/JITS
    predicted_vs_ids: list[str]  # In rank order

    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0

    error: str | None = None

    def compute(self) -> None:
        """Compute all metrics from predicted_vs_ids vs gold_vs_ids."""
        gold = set(self.gold_vs_ids)
        preds = self.predicted_vs_ids

        self.hit_at_1 = bool(preds[:1] and preds[0] in gold)
        self.hit_at_3 = any(p in gold for p in preds[:3])
        self.hit_at_5 = any(p in gold for p in preds[:5])

        top5 = preds[:5]
        if top5:
            hits5 = sum(1 for p in top5 if p in gold)
            self.precision_at_5 = hits5 / len(top5)
            self.recall_at_5 = hits5 / len(gold) if gold else 0.0
        else:
            self.precision_at_5 = 0.0
            self.recall_at_5 = 0.0

        # MRR
        self.mrr = 0.0
        for rank, pred in enumerate(preds[:10], start=1):
            if pred in gold:
                self.mrr = 1.0 / rank
                break

        # NDCG@5
        self.ndcg_at_5 = self._ndcg(preds[:5], gold)

    @staticmethod
    def _ndcg(preds: list[str], gold: set[str], k: int = 5) -> float:
        """Compute NDCG@k with binary relevance."""
        def dcg(ranked: list[str]) -> float:
            return sum(
                int(p in gold) / math.log2(i + 2) for i, p in enumerate(ranked)
            )

        ideal_hits = min(len(gold), k)
        ideal_ranked = list(gold)[:ideal_hits] + [""] * (k - ideal_hits)
        idcg = dcg(ideal_ranked)
        if idcg == 0:
            return 0.0
        return dcg(preds) / idcg


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across the full evaluation set."""

    total_cards: int = 0
    evaluated_cards: int = 0
    failed_cards: int = 0

    avg_hit_at_1: float = 0.0
    avg_hit_at_3: float = 0.0
    avg_hit_at_5: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg_at_5: float = 0.0

    per_card_results: list[CardEvaluation] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Compute mean metrics from per-card results."""
        valid = [r for r in self.per_card_results if r.error is None]
        n = len(valid)
        if n == 0:
            return

        self.evaluated_cards = n
        self.avg_hit_at_1 = sum(r.hit_at_1 for r in valid) / n
        self.avg_hit_at_3 = sum(r.hit_at_3 for r in valid) / n
        self.avg_hit_at_5 = sum(r.hit_at_5 for r in valid) / n
        self.avg_precision_at_5 = sum(r.precision_at_5 for r in valid) / n
        self.avg_recall_at_5 = sum(r.recall_at_5 for r in valid) / n
        self.avg_mrr = sum(r.mrr for r in valid) / n
        self.avg_ndcg_at_5 = sum(r.ndcg_at_5 for r in valid) / n

    def summary_dict(self) -> dict[str, Any]:
        return {
            "total_cards": self.total_cards,
            "evaluated_cards": self.evaluated_cards,
            "failed_cards": self.failed_cards,
            "hit@1": round(self.avg_hit_at_1, 4),
            "hit@3": round(self.avg_hit_at_3, 4),
            "hit@5": round(self.avg_hit_at_5, 4),
            "precision@5": round(self.avg_precision_at_5, 4),
            "recall@5": round(self.avg_recall_at_5, 4),
            "mrr": round(self.avg_mrr, 4),
            "ndcg@5": round(self.avg_ndcg_at_5, 4),
        }

    def print_report(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Value Stream RAG – Evaluation Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for k, v in self.summary_dict().items():
            table.add_row(k, str(v))

        console.print(table)


class RecommendationEvaluator:
    """
    Offline evaluator: runs the recommendation pipeline against historical
    idea cards with known ground-truth Value Stream mappings.
    """

    def __init__(self, recommendation_service) -> None:
        self._svc = recommendation_service

    def evaluate_dataset(
        self,
        eval_records: list[dict],
    ) -> EvaluationReport:
        """
        Evaluate the system against a list of historical records.

        Each record must have:
          - file_path: str
          - card_id: str (optional)
          - gold_vs_ids: list[str]

        Returns an EvaluationReport with aggregated metrics.
        """
        report = EvaluationReport(total_cards=len(eval_records))
        per_card: list[CardEvaluation] = []

        for record in eval_records:
            card_eval = CardEvaluation(
                card_id=record.get("card_id", ""),
                file_path=record["file_path"],
                gold_vs_ids=record["gold_vs_ids"],
                predicted_vs_ids=[],
            )

            try:
                result: RecommendationResult = self._svc.recommend_from_file(
                    file_path=record["file_path"],
                    domain_hint=record.get("domain"),
                )
                card_eval.predicted_vs_ids = [
                    vs.id for vs in result.recommended_value_streams
                ]
                card_eval.compute()
                logger.info(
                    "Evaluated %s | hit@5=%s | mrr=%.3f",
                    record["file_path"],
                    card_eval.hit_at_5,
                    card_eval.mrr,
                )
            except Exception as exc:
                logger.error("Evaluation failed for %s: %s", record["file_path"], exc)
                card_eval.error = str(exc)
                report.failed_cards += 1

            per_card.append(card_eval)

        report.per_card_results = per_card
        report.compute_aggregates()
        return report
