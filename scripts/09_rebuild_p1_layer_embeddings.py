#!/usr/bin/env python
"""Cache every MERT hidden layer using production-uniform segmentation."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vocaptest.audio.segment import split_segments
from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.models.mert_embedder import MERTEmbedder
from vocaptest.utils.config import load_config
from vocaptest.utils.paths import project_root

from importlib.util import module_from_spec, spec_from_file_location


def load_rebuild_helpers(root: Path):
    path = root / "scripts/07b_rebuild_curated_embeddings.py"
    spec = spec_from_file_location("p0_rebuild", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, records: list[EmbeddingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.segment_id):
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    root = project_root()
    helpers = load_rebuild_helpers(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default=root / "data/processed/curated/mert_95/song_decisions.jsonl",
        type=Path,
    )
    parser.add_argument("--audio-root", default=root / "data/audio", type=Path)
    parser.add_argument(
        "--embedding-output",
        default=root / "data/processed/embeddings/mert_95_p1_layers",
        type=Path,
    )
    parser.add_argument(
        "--manifest-output",
        default=root / "data/processed/curated/mert_95_p1/segments.jsonl",
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.decisions = args.decisions.resolve()
    args.audio_root = args.audio_root.resolve()
    args.embedding_output = args.embedding_output.resolve()
    args.manifest_output = args.manifest_output.resolve()

    cfg = load_config(root / "configs/default.yaml", root / "configs/model_mert.yaml")
    decisions = helpers.load_decisions(args.decisions)
    audio_paths = {
        item["song_id"]: helpers.find_audio_file(
            args.audio_root, item["producer_slug"], item["song_id"]
        )
        for item in decisions
    }

    existing = helpers.load_existing_records(args.manifest_output)
    existing_by_song: dict[str, list[EmbeddingRecord]] = {}
    for record in existing:
        existing_by_song.setdefault(record.song_id, []).append(record)
    complete_songs = {
        song_id for song_id, records in existing_by_song.items()
        if records and all((root / record.embedding_path).exists() for record in records)
    }
    all_records = [
        record for record in existing if record.song_id in complete_songs
    ]
    pending = [
        decision for decision in decisions
        if decision["song_id"] not in complete_songs
    ]

    args.embedding_output.mkdir(parents=True, exist_ok=True)
    embedder = None
    if pending:
        embedder = MERTEmbedder(
            model_name=cfg.model.get("hf_name", "m-a-p/MERT-v1-95M"),
            device=args.device,
            layer_strategy=cfg.model.get("layer_strategy", "mean_last_hidden"),
        )

    sample_rate = cfg.audio.get("sample_rate", 24000)
    for decision in tqdm(decisions, desc="Caching all MERT layers"):
        song_id = decision["song_id"]
        if song_id in complete_songs:
            continue
        wav = helpers.load_audio(audio_paths[song_id], sample_rate)
        segments = split_segments(
            wav,
            sample_rate,
            segment_seconds=cfg.audio.get("segment_seconds", 20.0),
            hop_seconds=cfg.audio.get("hop_seconds", 10.0),
            min_rms_db=cfg.audio.get("min_rms_db", -45.0),
            max_segments=cfg.audio.get("max_segments_per_song", 12),
            selection_strategy=cfg.audio.get("segment_selection", "uniform"),
        )
        chunks = [
            wav[item["start_sample"]:item["end_sample"]]
            for item in segments
        ]
        layer_embeddings: list[np.ndarray] = []
        for start in range(0, len(chunks), args.batch_size):
            assert embedder is not None
            layer_embeddings.extend(embedder.embed_batch_layers(
                chunks[start:start + args.batch_size],
                sample_rate,
            ))

        song_records = []
        for index, (segment, embedding) in enumerate(zip(segments, layer_embeddings)):
            start_ms = int(round(segment["start_sec"] * 1000))
            segment_id = f"{song_id}_p1s{index:03d}_{start_ms:07d}"
            output_path = args.embedding_output / f"{segment_id}.npy"
            np.save(output_path, np.asarray(embedding, dtype=np.float32))
            song_records.append(EmbeddingRecord(
                segment_id=segment_id,
                song_id=song_id,
                producer_slug=decision["producer_slug"],
                model_backend="mert_95_all_layers",
                embedding_path=output_path.relative_to(root).as_posix(),
                embedding_dim=int(embedding.shape[-1]),
                work_id=decision["work_id"],
                recording_id=song_id,
                title=decision["title"],
                layer_count=int(embedding.shape[0]),
            ))
        all_records.extend(song_records)
        write_manifest(args.manifest_output, all_records)

    layer_count = all_records[0].layer_count if all_records else None
    summary = {
        "songs": len({record.song_id for record in all_records}),
        "segments": len(all_records),
        "layers": layer_count,
        "embedding_dim": all_records[0].embedding_dim if all_records else None,
        "batch_size": args.batch_size,
        "device": args.device,
        "selection_strategy": cfg.audio.get("segment_selection", "uniform"),
    }
    with open(args.manifest_output.parent / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
