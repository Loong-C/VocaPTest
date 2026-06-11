#!/usr/bin/env python
"""Extract cached song-level MIR features for the curated P1 dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed

from vocaptest.features.mir_features import extract_mir_features, mir_feature_names
from vocaptest.utils.paths import project_root

from importlib.util import module_from_spec, spec_from_file_location


def load_helpers(root: Path):
    path = root / "scripts/07b_rebuild_curated_embeddings.py"
    spec = spec_from_file_location("p0_rebuild", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = project_root()
    helpers = load_helpers(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default=root / "data/processed/curated/mert_95/song_decisions.jsonl",
        type=Path,
    )
    parser.add_argument("--audio-root", default=root / "data/audio", type=Path)
    parser.add_argument(
        "--output",
        default=root / "data/processed/features/p1_mir_features.jsonl",
        type=Path,
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    decisions = helpers.load_decisions(args.decisions)

    def process(decision: dict) -> dict:
        path = helpers.find_audio_file(
            args.audio_root,
            decision["producer_slug"],
            decision["song_id"],
        )
        wav = helpers.load_audio(path, args.sample_rate)
        values = extract_mir_features(wav, args.sample_rate)
        return {
            "song_id": decision["song_id"],
            "work_id": decision["work_id"],
            "producer_slug": decision["producer_slug"],
            "feature_names": mir_feature_names(),
            "values": values.tolist(),
        }

    rows = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(process)(decision) for decision in decisions
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["song_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "songs": len(rows),
        "features": len(mir_feature_names()),
        "jobs": args.jobs,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
