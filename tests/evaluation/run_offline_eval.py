"""
Offline evaluation runner script.

Usage:
    python tests/evaluation/run_offline_eval.py \
        --eval-manifest data/eval_manifest.json \
        --output-dir data/eval_results

The eval manifest is a JSON file containing a list of records:
  [
    {
      "card_id": "JIRA-1234",
      "file_path": "data/historical_ppts/idea_card_1234.pptx",
      "gold_vs_ids": ["VS001", "VS003"],
      "domain": "Supply Chain"
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline Value Stream RAG evaluation")
    parser.add_argument("--eval-manifest", required=True, help="Path to eval manifest JSON")
    parser.add_argument("--output-dir", default="data/eval_results", help="Output directory")
    args = parser.parse_args()

    # Load evaluation manifest
    manifest_path = Path(args.eval_manifest)
    if not manifest_path.exists():
        print(f"ERROR: Eval manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        eval_records = json.load(f)

    print(f"Loaded {len(eval_records)} evaluation records")

    # Bootstrap services
    from src.config.settings import get_settings
    from src.services.container import ServiceContainer
    from src.services.recommendation_service import RecommendationService
    from src.evaluation.evaluator import RecommendationEvaluator

    settings = get_settings()
    container = ServiceContainer(settings)
    recommendation_svc = RecommendationService(
        recommendation_graph=container.recommendation_graph,
        upload_dir=settings.upload_dir,
    )
    evaluator = RecommendationEvaluator(recommendation_svc)

    print("Running evaluation...")
    report = evaluator.evaluate_dataset(eval_records)

    # Print summary
    report.print_report()

    # Save full report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{timestamp}.json"

    report_data = {
        "summary": report.summary_dict(),
        "per_card": [
            {
                "card_id": r.card_id,
                "file_path": r.file_path,
                "gold_vs_ids": r.gold_vs_ids,
                "predicted_vs_ids": r.predicted_vs_ids,
                "hit@1": r.hit_at_1,
                "hit@3": r.hit_at_3,
                "hit@5": r.hit_at_5,
                "precision@5": r.precision_at_5,
                "recall@5": r.recall_at_5,
                "mrr": r.mrr,
                "ndcg@5": r.ndcg_at_5,
                "error": r.error,
            }
            for r in report.per_card_results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nFull report saved to: {report_path}")

    # Return exit code 1 if Hit@5 is below acceptable threshold
    if report.avg_hit_at_5 < 0.5:
        print(
            f"WARNING: Hit@5 = {report.avg_hit_at_5:.3f} is below the 0.5 acceptance threshold"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
