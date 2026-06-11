#!/usr/bin/env python
"""Build producer profiles from embedding records."""
import argparse
import json
from pathlib import Path

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.retrieval.build_profiles import build_producer_profiles
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Build producer profiles")
    parser.add_argument("--embeddings", required=True, help="Path to embedding manifest JSONL")
    parser.add_argument("--clusters", type=int, default=5, help="KMeans clusters per producer")
    parser.add_argument("--output", required=True, help="Output pickle file path")
    args = parser.parse_args()

    # Load records
    records = []
    with open(args.embeddings, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EmbeddingRecord(**json.loads(line)))

    logger.info("Loaded %d embedding records", len(records))

    profiles = build_producer_profiles(
        records,
        num_clusters=args.clusters,
        output_path=args.output,
    )

    logger.info(
        "Profiles built: %d producers, backend=%s",
        len(profiles["producers"]), profiles["backend"],
    )


if __name__ == "__main__":
    main()
