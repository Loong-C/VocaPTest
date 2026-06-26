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
    download_media_audio,
    media_source_from_item,
    read_media_metadata,
    source_reason,
    source_key,
    source_url,
    validate_vocadb_pv,
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
        source_key(*media_source_from_item(item))
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
    seen_sources: set[tuple[str, str]] = set()
    for item in additions:
        slug = item["producer_slug"]
        source_service, source_id = media_source_from_item(item)
        if slug not in producers:
            raise ValueError(f"Unknown producer slug: {slug}")
        source_tuple = (source_service, source_id)
        if source_tuple in seen_sources:
            raise ValueError(f"Duplicate source in additions: {source_tuple}")
        seen_sources.add(source_tuple)

        song_id = source_key(source_service, source_id)
        url = source_url(source_service, source_id)
        output_path = args.audio_root / slug / f"{song_id}.mp3"
        if song_id in existing_accepted and output_path.exists():
            new_records.append(existing_accepted[song_id])
            print(f"reused {slug}/{item['title']} ({source_id})")
            continue

        source_kind = item.get("source_kind", "official_upload")
        vocadb = validate_vocadb_pv(
            song_id=int(item["vocadb_song_id"]),
            source_service=source_service,
            source_id=source_id,
            artist_id=int(producers[slug]["vocadb_artist_id"]),
            session=session,
            allowed_pv_types=(
                ("Reprint",)
                if source_kind == "vocadb_reprint"
                else ("Original",)
            ),
        )
        metadata = read_media_metadata(command, url)
        channel_id = metadata.get("channel_id") or metadata.get("uploader_id")
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
            download_media_audio(
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
            "source_service": source_service,
            "source_id": source_id,
            "source_channel_id": channel_id,
            "source_kind": source_kind,
            "vocadb_song_id": item["vocadb_song_id"],
            **vocadb,
        })
        print(f"verified {slug}/{item['title']} ({source_id})")

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
