#!/usr/bin/env python
"""Validate and download a strictly held-out evaluation catalog."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import requests
import yaml

from vocaptest.data.catalog_sources import (
    download_youtube_audio,
    read_youtube_metadata,
    source_reason,
    validate_vocadb_original,
    yt_dlp_command,
)
from vocaptest.utils.paths import project_root


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_yaml_songs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("songs", [])


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "configs" / "frozen_test_catalog.yaml",
    )
    parser.add_argument(
        "--training-decisions",
        type=Path,
        default=(
            root
            / "data"
            / "processed"
            / "curated"
            / "mert_95"
            / "song_decisions.jsonl"
        ),
    )
    parser.add_argument(
        "--training-catalog",
        type=Path,
        default=root / "configs" / "training_catalog_additions.yaml",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=root / "data" / "frozen_test_audio",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=root / "data" / "processed" / "frozen_test" / "catalog.jsonl",
    )
    parser.add_argument("--category", default="frozen_test")
    parser.add_argument("--expected-per-class", type=int, default=4)
    parser.add_argument(
        "--exclude-catalog",
        action="append",
        default=[],
        type=Path,
        help="Additional YAML song catalogs that must not overlap this catalog.",
    )
    parser.add_argument(
        "--ffmpeg-location",
        type=Path,
        help="Directory containing ffmpeg and ffprobe when they are not on PATH.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Only process selected producer slugs; may be repeated.",
    )
    args = parser.parse_args()

    songs = load_yaml_songs(args.catalog)
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        producers = {
            item["slug"]: item
            for item in (yaml.safe_load(handle) or {}).get("producers", [])
        }

    counts = Counter(item["producer_slug"] for item in songs)
    missing = set(producers).difference(counts)
    unexpected = set(counts).difference(producers)
    if missing or unexpected:
        raise ValueError(
            f"Frozen catalog class mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    wrong_counts = {
        slug: count for slug, count in counts.items()
        if count != args.expected_per_class
    }
    if wrong_counts:
        raise ValueError(
            f"Each producer needs exactly {args.expected_per_class} "
            f"{args.category} songs: {wrong_counts}"
        )

    training = load_jsonl(args.training_decisions)
    configured_training = load_yaml_songs(args.training_catalog)
    training_youtube = {
        item["song_id"].removeprefix("youtube_")
        for item in training
        if item.get("status") == "accepted"
        and item["song_id"].startswith("youtube_")
    }
    training_vocadb = {
        int(item["vocadb_song_id"])
        for item in training
        if item.get("status") == "accepted"
        and item.get("vocadb_song_id") is not None
    }
    training_youtube.update(
        item["youtube_id"] for item in configured_training
    )
    training_vocadb.update(
        int(item["vocadb_song_id"]) for item in configured_training
    )
    for catalog_path in args.exclude_catalog:
        excluded = load_yaml_songs(catalog_path)
        training_youtube.update(item["youtube_id"] for item in excluded)
        training_vocadb.update(int(item["vocadb_song_id"]) for item in excluded)
    frozen_youtube = [item["youtube_id"] for item in songs]
    frozen_vocadb = [int(item["vocadb_song_id"]) for item in songs]
    if len(frozen_youtube) != len(set(frozen_youtube)):
        raise ValueError("Duplicate YouTube IDs in frozen test catalog")
    if len(frozen_vocadb) != len(set(frozen_vocadb)):
        raise ValueError("Duplicate VocaDB song IDs in frozen test catalog")
    youtube_overlap = training_youtube.intersection(frozen_youtube)
    vocadb_overlap = training_vocadb.intersection(frozen_vocadb)
    if youtube_overlap or vocadb_overlap:
        raise ValueError(
            f"Train/frozen overlap: YouTube={sorted(youtube_overlap)}, "
            f"VocaDB={sorted(vocadb_overlap)}"
        )

    requested = set(args.slug)
    if requested:
        unknown = requested.difference(producers)
        if unknown:
            raise ValueError(f"Unknown producer slug(s): {sorted(unknown)}")
        songs = [
            item for item in songs
            if item["producer_slug"] in requested
        ]

    command = yt_dlp_command(root)
    session = requests.Session()
    session.headers["User-Agent"] = "VocaPTest/0.1 frozen catalog validation"
    existing = {
        item["song_id"]: item
        for item in load_jsonl(args.manifest_output)
    }
    records = dict(existing)
    for item in songs:
        slug = item["producer_slug"]
        video_id = item["youtube_id"]
        song_id = f"youtube_{video_id}"
        url = f"https://www.youtube.com/watch?v={video_id}"
        source_kind = item.get("source_kind", "official_upload")
        vocadb = validate_vocadb_original(
            song_id=int(item["vocadb_song_id"]),
            youtube_id=video_id,
            artist_id=int(producers[slug]["vocadb_artist_id"]),
            session=session,
            allowed_pv_types=(
                ("Reprint",)
                if source_kind == "vocadb_reprint"
                else ("Other",)
                if source_kind == "vocadb_other_pv"
                else ("Original",)
            ),
        )
        metadata = read_youtube_metadata(command, url)
        channel_id = metadata.get("channel_id")
        allowed_channels = set(item.get("allowed_channel_ids", []))
        if allowed_channels and channel_id not in allowed_channels:
            raise ValueError(
                f"{song_id} came from unexpected channel {channel_id}; "
                f"expected one of {sorted(allowed_channels)}"
            )
        duration = float(metadata.get("duration") or 0)
        if not 60 <= duration <= 600:
            raise ValueError(f"Unexpected duration for {song_id}: {duration}")

        output_path = args.audio_root / slug / f"{song_id}.mp3"
        if not args.skip_download and not output_path.exists():
            download_youtube_audio(
                command,
                url,
                output_path,
                args.ffmpeg_location,
            )
        if not output_path.exists():
            raise FileNotFoundError(f"Missing audio for {song_id}: {output_path}")

        records[song_id] = {
            "song_id": song_id,
            "producer_slug": slug,
            "title": item["title"],
            "status": "accepted",
            "category": args.category,
            "reason": source_reason(source_kind),
            "work_id": f"vocadb_song_{item['vocadb_song_id']}",
            "canonical_song_id": song_id,
            "segment_count": None,
            "source_url": url,
            "source_channel_id": channel_id,
            "source_kind": source_kind,
            "vocadb_song_id": int(item["vocadb_song_id"]),
            "duration_seconds": duration,
            **vocadb,
        }
        write_jsonl(
            args.manifest_output,
            sorted(
                records.values(),
                key=lambda record: (
                    record["producer_slug"],
                    record["title"].casefold(),
                ),
            ),
        )
        print(f"verified {args.category} {slug}/{item['title']} ({video_id})")

    print(json.dumps(
        {
            "songs": len(records),
            "classes": len({
                record["producer_slug"] for record in records.values()
            }),
            "songs_per_class": dict(sorted(Counter(
                record["producer_slug"] for record in records.values()
            ).items())),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
