#!/usr/bin/env python
"""Build an auditable, portable song-level dataset from legacy embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vocaptest.data.curation import (
    curate_embedding_records,
    load_curation_config,
    load_embedding_manifest,
    load_song_titles,
    write_curation_result,
)
from vocaptest.features.song_features import build_song_feature_matrix, save_song_features
from vocaptest.utils.paths import project_root


def find_near_duplicate_candidates(
    features: np.ndarray,
    metadata: list,
    threshold: float,
) -> list[dict]:
    candidates: list[dict] = []
    for left in range(len(metadata)):
        for right in range(left + 1, len(metadata)):
            if metadata[left].producer_slug != metadata[right].producer_slug:
                continue
            similarity = float(features[left] @ features[right])
            if similarity >= threshold:
                candidates.append({
                    "producer_slug": metadata[left].producer_slug,
                    "left_song_id": metadata[left].song_id,
                    "left_title": metadata[left].title,
                    "right_song_id": metadata[right].song_id,
                    "right_title": metadata[right].title,
                    "cosine_similarity": similarity,
                    "review_status": "pending_manual_review",
                })
    return sorted(candidates, key=lambda item: item["cosine_similarity"], reverse=True)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        default=root / "data/processed/embeddings/mert_95/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--titles",
        default=root / "data/processed/song_name_mapping.csv",
        type=Path,
    )
    parser.add_argument(
        "--config",
        default=root / "configs/dataset_curation.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=root / "data/processed/curated/mert_95",
        type=Path,
    )
    args = parser.parse_args()

    records = load_embedding_manifest(args.embeddings)
    titles = load_song_titles(args.titles)
    config = load_curation_config(args.config)
    result = curate_embedding_records(records, titles, config)
    write_curation_result(result, args.output)

    features, metadata = build_song_feature_matrix(result.records)
    save_song_features(features, metadata, args.output)

    threshold = config.get("curation", {}).get(
        "near_duplicate_review_threshold", 0.985
    )
    candidates = find_near_duplicate_candidates(features, metadata, threshold)
    with open(args.output / "near_duplicate_candidates.json", "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        **result.summary,
        "feature_shape": list(features.shape),
        "near_duplicate_candidates": len(candidates),
        "output_dir": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
