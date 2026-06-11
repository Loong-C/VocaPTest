#!/usr/bin/env python
"""YouTube-based search + download pipeline. Replaces VocaDB for MVP."""
import argparse
import json
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from datetime import datetime

import yaml

from vocaptest.utils.logging import setup_logging

logger = setup_logging()

YT_DLP = str(Path(__file__).resolve().parent.parent / "tools" / "yt-dlp.exe")


def search_youtube(producer_slug: str, display_name: str, max_results: int = 30) -> list[dict]:
    """Search YouTube for Vocaloid songs by a producer. Returns parsed metadata."""
    query = f"ytsearch{max_results}:{display_name} VOCALOID"
    logger.info("Searching: %s", query)

    result = subprocess.run(
        [YT_DLP, "--flat-playlist", "--dump-json", query],
        capture_output=True, text=True, timeout=120,
    )

    songs = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("_type") != "url":
            continue
        if item.get("ie_key") != "Youtube":
            continue

        # Basic filtering: skip very short (< 60s) or very long (> 600s)
        duration = item.get("duration") or 0
        if duration < 60 or duration > 600:
            continue

        songs.append({
            "id": item["id"],
            "url": item["url"],
            "title": item.get("title", ""),
            "duration": duration,
            "channel": item.get("channel", ""),
            "uploader": item.get("uploader", ""),
            "uploader_url": item.get("uploader_url", ""),
            "view_count": item.get("view_count", 0),
        })

    # Deduplicate by video ID
    seen = set()
    unique = []
    for s in songs:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)
    return unique


def download_audio(video_url: str, output_dir: Path, song_id: str) -> Path | None:
    """Download audio-only from YouTube video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{song_id}.%(ext)s")

    cmd = [
        YT_DLP,
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        video_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        last_line = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown error"
        logger.warning("Download failed for %s: %s", song_id, last_line[:120])
        return None

    # Find the downloaded file
    mp3_path = output_dir / f"{song_id}.mp3"
    if mp3_path.exists():
        return mp3_path
    return None


def compute_file_hash(path: Path) -> str:
    """SHA-256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="YouTube search + download pipeline")
    parser.add_argument("--producers", required=True, help="Path to producers.yaml")
    parser.add_argument("--audio-dir", required=True, help="Directory for downloaded audio")
    parser.add_argument("--output-dir", required=True, help="Output directory for metadata")
    parser.add_argument("--max-per-producer", type=int, default=15, help="Max songs to download per producer")
    parser.add_argument("--search-results", type=int, default=40, help="How many YouTube results to search")
    parser.add_argument("--download", action="store_true", help="Actually download audio")
    args = parser.parse_args()

    with open(args.producers, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_songs = []
    manifest_entries = []

    for producer in data["producers"]:
        slug = producer["slug"]
        display_name = producer["display_name"]
        producer_audio_dir = audio_dir / slug

        logger.info("=" * 60)
        logger.info("Producer: %s (%s)", display_name, slug)

        # Step 1: Search YouTube
        results = search_youtube(slug, display_name, max_results=args.search_results)
        logger.info("Found %d YouTube results after filtering", len(results))

        # Step 2: Convert to Song records (top N)
        songs_for_producer = []
        for i, r in enumerate(results[:args.max_per_producer]):
            song_id = f"youtube_{r['id']}"
            song = {
                "song_id": song_id,
                "producer_slug": slug,
                "title": r["title"],
                "publish_date": None,
                "source_urls": [r["url"]],
                "vocalists": [],
                "tags": ["VOCALOID"],
                "is_cover": False,
                "is_remix": False,
                "is_collaboration": False,
                "status": "accepted",
                "status_reason": None,
                "local_audio_path": None,
            }
            songs_for_producer.append(song)

        # Step 3: Download audio
        if args.download:
            for song in songs_for_producer:
                url = song["source_urls"][0]
                logger.info("  [%s] Downloading: %s", slug, song["title"][:60])
                path = download_audio(url, producer_audio_dir, song["song_id"])
                if path:
                    song["local_audio_path"] = str(path.resolve())
                    manifest_entries.append({
                        "file_hash": compute_file_hash(path),
                        "path": str(path.resolve()),
                        "duration_sec": None,
                        "sample_rate": None,
                        "channels": None,
                        "source_url": url,
                        "producer_slug": slug,
                        "song_id": song["song_id"],
                    })
                else:
                    song["status"] = "rejected"
                    song["status_reason"] = "download_failed"

        all_songs.extend(songs_for_producer)

    # Save metadata
    jsonl_path = output_dir / "youtube_songs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in all_songs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Saved %d songs to %s", len(all_songs), jsonl_path)

    # Save manifest
    if manifest_entries:
        manifest_path = output_dir / "audio_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for e in manifest_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        logger.info("Saved %d manifest entries to %s", len(manifest_entries), manifest_path)

    # Summary
    downloaded = sum(1 for s in all_songs if s.get("local_audio_path"))
    logger.info("Summary: %d producers, %d songs, %d downloaded",
                 len(data["producers"]), len(all_songs), downloaded)


if __name__ == "__main__":
    main()
