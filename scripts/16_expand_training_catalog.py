#!/usr/bin/env python
"""Download vetted catalog additions and merge them into curation decisions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from vocaptest.utils.paths import project_root


def yt_dlp_command(root: Path) -> list[str]:
    bundled = root / "tools" / "yt-dlp.exe"
    if bundled.exists():
        return [str(bundled)]
    return [sys.executable, "-m", "yt_dlp"]


def read_metadata(command: list[str], url: str) -> dict:
    result = subprocess.run(
        command + ["--skip-download", "--no-warnings", "--dump-single-json", url],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Metadata failed: {url}")
    return json.loads(result.stdout)


def download_audio(
    command: list[str],
    url: str,
    output_path: Path,
    ffmpeg_location: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_args = (
        ["--ffmpeg-location", str(ffmpeg_location)]
        if ffmpeg_location
        else []
    )
    result = subprocess.run(
        command
        + [
            "-f",
            "bestaudio[ext=m4a]/bestaudio",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "192K",
            "--no-playlist",
            "--no-warnings",
        ]
        + ffmpeg_args
        + [
            "-o",
            str(output_path.with_suffix(".%(ext)s")),
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if result.returncode != 0 or not output_path.exists():
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(
            detail[-1] if detail else f"Audio download failed: {url}"
        )


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
    args = parser.parse_args()

    with open(args.catalog, "r", encoding="utf-8") as handle:
        additions = (yaml.safe_load(handle) or {}).get("songs", [])
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        producer_slugs = {
            item["slug"]
            for item in (yaml.safe_load(handle) or {}).get("producers", [])
        }

    command = yt_dlp_command(root)
    new_records = []
    seen_video_ids: set[str] = set()
    for item in additions:
        slug = item["producer_slug"]
        video_id = item["youtube_id"]
        if slug not in producer_slugs:
            raise ValueError(f"Unknown producer slug: {slug}")
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate YouTube ID in additions: {video_id}")
        seen_video_ids.add(video_id)

        song_id = f"youtube_{video_id}"
        url = f"https://www.youtube.com/watch?v={video_id}"
        metadata = read_metadata(command, url)
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
            download_audio(command, url, output_path, args.ffmpeg_location)
        if not output_path.exists():
            raise FileNotFoundError(f"Missing audio for {song_id}: {output_path}")

        source_kind = item.get("source_kind", "official_upload")
        reason = {
            "official_upload": (
                "VocaDB original work with verified official YouTube source"
            ),
            "vocadb_reprint": (
                "VocaDB original work with a VocaDB-listed YouTube reprint"
            ),
        }.get(source_kind)
        if reason is None:
            raise ValueError(
                f"Unsupported source_kind for {song_id}: {source_kind}"
            )

        new_records.append({
            "song_id": song_id,
            "producer_slug": slug,
            "title": item["title"],
            "status": "accepted",
            "category": "vetted_catalog_expansion",
            "reason": "VocaDB original work with verified official YouTube source",
            "work_id": f"vocadb_song_{item['vocadb_song_id']}",
            "canonical_song_id": song_id,
            "segment_count": None,
            "source_url": url,
            "source_channel_id": channel_id,
            "source_kind": source_kind,
            "vocadb_song_id": item["vocadb_song_id"],
        })
        print(f"verified {slug}/{item['title']} ({video_id})")

    decisions = load_jsonl(args.decisions)
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
