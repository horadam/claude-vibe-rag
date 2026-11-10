"""CLI entry point for corpus ingestion."""

import argparse

from src.ingest import export_index, ingest_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the handbook index.")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="rewrite the JSON index from the existing Chroma collection, without embedding",
    )
    args = parser.parse_args()
    if args.export_only:
        export_index()
    else:
        ingest_corpus()
