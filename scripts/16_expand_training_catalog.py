#!/usr/bin/env python
"""Download vetted catalog additions and merge them into curation decisions."""
from __future__ import annotations

import argparse
import json
import sys
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
    with open(path, "r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def write_jsonl(path: Path, records: list[dict]) -> None:
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
        default=root / "configs" / "training_catalog_additions.yaml",
    )
    parser.add_argument(
        "--decisions",
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
        "--audio-root",
        type=Path,
        default=root / "data" / "audio",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--ffmpeg-location",
        type=Path,
        help="Directory containing ffmpeg and ffprobe when they are not on PATH.",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Only process selected producer slugs; may be repeated.",
    )
    args = parser.parse_args()

    with open(args.catalog, "r", encoding="utf-8") as handle:
        all_additions = (yaml.safe_load(handle) or {}).get("songs", [])
    additions = list(all_additions)
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        producers = {
            item["slug"]: item
            for item in (yaml.safe_load(handle) or {}).get("producers", [])
        }
    if args.slug:
        requested = set(args.slug)
        unknown = requested.difference(producers)
        if unknown:
            raise ValueError(f"Unknown producer slug(s): {sorted(unknown)}")
        additions = [
            item for item in additions
            if item["producer_slug"] in requested
        ]
    else:
        requested = set()

    existing_decisions = load_jsonl(args.decisions) if args.decisions.exists() else []
    configured_song_ids = {
        f"youtube_{item['youtube_id']}"
        for item in all_additions
    }
    retained_decisions = [
        item for item in existing_decisions
        if item.get("category") != "vetted_catalog_expansion"
        or item["song_id"] in configured_song_ids
        or (requested and item.get("producer_slug") not in requested)
    ]
    existing_accepted = {
        item["song_id"]: item
        for item in retained_decisions
        if item.get("status") == "accepted"
    }

    command = yt_dlp_command(root)
    session = requests.Session()
    session.headers["User-Agent"] = "VocaPTest/0.1 catalog validation"
    new_records = []
    seen_video_ids: set[str] = set()
    for item in additions:
        slug = item["producer_slug"]
        video_id = item["youtube_id"]
        if slug not in producers:
            raise ValueError(f"Unknown producer slug: {slug}")
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate YouTube ID in additions: {video_id}")
        seen_video_ids.add(video_id)

        song_id = f"youtube_{video_id}"
        url = f"https://www.youtube.com/watch?v={video_id}"
        output_path = args.audio_root / slug / f"{song_id}.mp3"
        if song_id in existing_accepted and output_path.exists():
            new_records.append(existing_accepted[song_id])
            print(f"reused {slug}/{item['title']} ({video_id})")
            continue

        source_kind = item.get("source_kind", "official_upload")
        vocadb = validate_vocadb_original(
            song_id=int(item["vocadb_song_id"]),
            youtube_id=video_id,
            artist_id=int(producers[slug]["vocadb_artist_id"]),
            session=session,
            allowed_pv_types=(
                ("Reprint",)
                if source_kind == "vocadb_reprint"
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

        if not args.skip_download and not output_path.exists():
            download_youtube_audio(
                command,
                url,
                output_path,
                args.ffmpeg_location,
            )
        if not output_path.exists():
            raise FileNotFoundError(f"Missing audio for {song_id}: {output_path}")

        reason = source_reason(source_kind)

        new_records.append({
            "song_id": song_id,
            "producer_slug": slug,
            "title": item["title"],
            "status": "accepted",
            "category": "vetted_catalog_expansion",
            "reason": reason,
            "work_id": f"vocadb_song_{item['vocadb_song_id']}",
            "canonical_song_id": song_id,
            "segment_count": None,
            "source_url": url,
            "source_channel_id": channel_id,
            "source_kind": source_kind,
            "vocadb_song_id": item["vocadb_song_id"],
            **vocadb,
        })
        print(f"verified {slug}/{item['title']} ({video_id})")

    decisions = retained_decisions
    existing_ids = {item["song_id"] for item in decisions}
    duplicates = existing_ids.intersection(item["song_id"] for item in new_records)
    if duplicates:
        decisions = [
            item
            for item in decisions
            if item["song_id"] not in duplicates
        ]
    decisions.extend(new_records)
    write_jsonl(args.decisions, decisions)
    print(
        json.dumps(
            {
                "additions": len(new_records),
                "accepted_total": sum(
                    item.get("status") == "accepted"
                    for item in decisions
                ),
                "decisions_total": len(decisions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
