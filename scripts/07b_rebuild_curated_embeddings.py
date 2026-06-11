#!/usr/bin/env python
"""Rebuild curated embeddings with the exact segmentation used by the API."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample
from tqdm import tqdm

from vocaptest.audio.segment import split_segments
from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.models.mert_embedder import MERTEmbedder
from vocaptest.utils.config import load_config
from vocaptest.utils.paths import project_root


def load_decisions(path: Path) -> list[dict]:
    decisions = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            decision = json.loads(line)
            if decision["status"] == "accepted":
                decisions.append(decision)
    return decisions


def find_audio_file(audio_root: Path, producer_slug: str, song_id: str) -> Path:
    matches = [
        path for path in (audio_root / producer_slug).glob(f"{song_id}.*")
        if path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one audio file for {producer_slug}/{song_id}, found {matches}"
        )
    return matches[0]


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    wav, input_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if input_rate != sample_rate:
        target_length = int(len(wav) * sample_rate / input_rate)
        wav = scipy_resample(wav, target_length).astype(np.float32)
    return wav


def load_existing_records(path: Path) -> list[EmbeddingRecord]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [
            EmbeddingRecord(**json.loads(line))
            for line in handle
            if line.strip()
        ]


def write_manifest(path: Path, records: list[EmbeddingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.segment_id):
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default=root / "data/processed/curated/mert_95/song_decisions.jsonl",
        type=Path,
    )
    parser.add_argument("--audio-root", default=root / "data/audio", type=Path)
    parser.add_argument(
        "--embedding-output",
        default=root / "data/processed/embeddings/mert_95_p0_20s",
        type=Path,
    )
    parser.add_argument(
        "--manifest-output",
        default=root / "data/processed/curated/mert_95_p0_20s/segments.jsonl",
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(
        root / "configs/default.yaml",
        root / "configs/model_mert.yaml",
    )
    sample_rate = cfg.audio.get("sample_rate", 24000)
    segment_seconds = cfg.audio.get("segment_seconds", 20.0)
    hop_seconds = cfg.audio.get("hop_seconds", 10.0)
    min_rms_db = cfg.audio.get("min_rms_db", -45.0)
    max_segments = cfg.audio.get("max_segments_per_song", 12)
    selection_strategy = cfg.audio.get("segment_selection", "uniform")

    decisions = load_decisions(args.decisions)
    if len(decisions) != 174:
        raise ValueError(f"Expected 174 accepted songs from P0 curation, found {len(decisions)}")
    audio_paths = {
        item["song_id"]: find_audio_file(
            args.audio_root, item["producer_slug"], item["song_id"]
        )
        for item in decisions
    }

    args.embedding_output.mkdir(parents=True, exist_ok=True)
    existing_records = [] if args.no_resume else load_existing_records(args.manifest_output)
    existing_by_song: dict[str, list[EmbeddingRecord]] = {}
    for record in existing_records:
        existing_by_song.setdefault(record.song_id, []).append(record)
    complete_songs = {
        song_id for song_id, records in existing_by_song.items()
        if records and all((root / record.embedding_path).exists() for record in records)
    }
    all_records = [
        record for record in existing_records if record.song_id in complete_songs
    ]

    pending_decisions = [
        decision for decision in decisions
        if decision["song_id"] not in complete_songs
    ]
    embedder = None
    if pending_decisions:
        embedder = MERTEmbedder(
            model_name=cfg.model.get("hf_name", "m-a-p/MERT-v1-95M"),
            device=args.device,
            layer_strategy=cfg.model.get("layer_strategy", "mean_last_hidden"),
        )

    for decision in tqdm(decisions, desc="Rebuilding song embeddings"):
        song_id = decision["song_id"]
        if song_id in complete_songs:
            continue

        wav = load_audio(audio_paths[song_id], sample_rate)
        segment_info = split_segments(
            wav,
            sample_rate,
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            min_rms_db=min_rms_db,
            max_segments=max_segments,
            selection_strategy=selection_strategy,
        )
        if not segment_info:
            raise ValueError(f"No valid segments for {song_id}")

        chunks = [
            wav[item["start_sample"]:item["end_sample"]]
            for item in segment_info
        ]
        embeddings: list[np.ndarray] = []
        for start in range(0, len(chunks), args.batch_size):
            assert embedder is not None
            embeddings.extend(embedder.embed_batch(
                chunks[start:start + args.batch_size],
                sample_rate,
            ))

        song_records: list[EmbeddingRecord] = []
        for index, (item, embedding) in enumerate(zip(segment_info, embeddings)):
            start_ms = int(round(item["start_sec"] * 1000))
            segment_id = f"{song_id}_p0s{index:03d}_{start_ms:07d}"
            embedding_path = args.embedding_output / f"{segment_id}.npy"
            np.save(embedding_path, np.asarray(embedding, dtype=np.float32))
            song_records.append(EmbeddingRecord(
                segment_id=segment_id,
                song_id=song_id,
                producer_slug=decision["producer_slug"],
                model_backend=embedder.backend_name,
                embedding_path=embedding_path.relative_to(root).as_posix(),
                embedding_dim=embedder.dim,
                work_id=decision["work_id"],
                recording_id=song_id,
                title=decision["title"],
            ))

        all_records.extend(song_records)
        write_manifest(args.manifest_output, all_records)

    embedding_dim = embedder.dim if embedder else all_records[0].embedding_dim
    embedding_backend = (
        embedder.backend_name if embedder else all_records[0].model_backend
    )
    summary = {
        "songs": len({record.song_id for record in all_records}),
        "segments": len(all_records),
        "embedding_dim": embedding_dim,
        "embedding_backend": embedding_backend,
        "segmentation": {
            "sample_rate": sample_rate,
            "segment_seconds": segment_seconds,
            "hop_seconds": hop_seconds,
            "min_rms_db": min_rms_db,
            "max_segments_per_song": max_segments,
            "selection_strategy": selection_strategy,
        },
        "batch_size": args.batch_size,
        "device": args.device,
    }
    summary_path = args.manifest_output.parent / "rebuild_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
