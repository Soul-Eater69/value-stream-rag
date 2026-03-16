"""
CLI script for batch ingestion of historical idea-card PPTs.

Usage:
    # Ingest from a directory using a JSON manifest
    python scripts/ingest_historical.py \
        --manifest data/historical_ppts/manifest.json

    # Ingest a single file
    python scripts/ingest_historical.py \
        --file data/historical_ppts/jira_1234.pptx \
        --vs-ids VS001 VS007 \
        --jira JIRA-1234

Manifest format (JSON):
    [
      {
        "file_path": "data/historical_ppts/jira_1234.pptx",
        "mapped_value_stream_ids": ["VS001", "VS007"],
        "mapped_value_stream_names": ["Order Management", "Inventory Control"],
        "jira_ticket": "JIRA-1234",
        "title": "Inventory Automation Initiative",
        "domain": "Supply Chain"
      }
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ingest historical idea-card PPTs into the RAG index"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Path to ingestion manifest JSON file")
    group.add_argument("--file", help="Single PPTX file to ingest")

    parser.add_argument("--vs-ids", nargs="+", help="Value Stream IDs (for --file mode)")
    parser.add_argument("--jira", help="JIRA ticket ID (for --file mode)", default=None)
    parser.add_argument("--domain", help="Business domain (for --file mode)", default="")
    parser.add_argument("--dry-run", action="store_true", help="Parse only; do not index")
    args = parser.parse_args()

    from src.config.logging_config import configure_logging
    from src.config.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level)

    # ── Build records
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
        with open(manifest_path) as f:
            raw_records = json.load(f)
    else:
        raw_records = [
            {
                "file_path": args.file,
                "mapped_value_stream_ids": args.vs_ids or [],
                "jira_ticket": args.jira,
                "domain": args.domain,
            }
        ]

    print(f"Loaded {len(raw_records)} record(s) for ingestion")

    if args.dry_run:
        print("DRY RUN: Parsing only (no indexing)")
        from src.ingestion.parser import PPTParser
        parser_obj = PPTParser()
        for r in raw_records:
            try:
                doc = parser_obj.parse(r["file_path"])
                print(f"  ✓ {r['file_path']}: {doc.slide_count} slides")
            except Exception as exc:
                print(f"  ✗ {r['file_path']}: {exc}")
        return

    # ── Build service container and run ingestion
    from src.services.container import ServiceContainer
    container = ServiceContainer(settings)
    pipeline = container.ingestion_pipeline

    from src.ingestion.pipeline import IngestionRecord
    records = [IngestionRecord(**r) for r in raw_records]
    results = pipeline.ingest_batch(records)

    # ── Print summary
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    print(f"\nIngestion complete: {succeeded} succeeded / {failed} failed")
    for result in results:
        status = "✓" if result.success else "✗"
        msg = f"chunks={result.chunk_count}" if result.success else f"error={result.error}"
        print(f"  {status} {result.file_path}: slides={result.slide_count}, {msg}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
