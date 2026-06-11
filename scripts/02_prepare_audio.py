#!/usr/bin/env python
"""Prepare audio: build song index, download audio, validate."""
import argparse
import json
from pathlib import Path

import yaml

from vocaptest.data.build_song_index import build_song_index, save_song_index
from vocaptest.data.download_audio import build_audio_manifest, download_audio
from vocaptest.data.validate_dataset import check_song_balance
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Prepare audio dataset")
    parser.add_argument("--metadata-dir", required=True, help="Dir with raw VocaDB JSONL files")
    parser.add_argument("--producers", required=True, help="Path to producers.yaml")
    parser.add_argument("--audio-dir", required=True, help="Directory for downloaded audio")
    parser.add_argument("--output-dir", required=True, help="Output directory for interim files")
    parser.add_argument("--download", action="store_true", help="Actually download audio files")
    parser.add_argument("--max-downloads", type=int, default=10, help="Max downloads per producer for MVP")
    args = parser.parse_args()

    with open(args.producers, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_songs = []
    for producer in data["producers"]:
        slug = producer["slug"]
        jsonl_path = metadata_dir / f"{slug}_songs.jsonl"
        if not jsonl_path.exists():
            logger.warning("Metadata not found for %s — skipping", slug)
            continue

        songs = build_song_index(jsonl_path, slug, artist_id=producer.get("vocadb_artist_id"))
        accepted = [s for s in songs if s.status == "accepted"]
        pending = [s for s in songs if s.status == "pending_review"]

        if pending:
            print(f"\n⚠ {slug}: {len(pending)} songs need manual review (pending_review)")
            print("  Check the metadata and manually update statuses before proceeding.")

        # Download audio for accepted songs (limited for MVP)
        if args.download:
            for song in accepted[:args.max_downloads]:
                if not song.source_urls:
                    continue
                url = song.source_urls[0]
                logger.info("Downloading: %s - %s", slug, song.title)
                downloaded = download_audio(
                    url,
                    audio_dir / slug,
                    output_template=song.song_id,
                )
                if downloaded:
                    song.local_audio_path = str(downloaded.resolve())
                else:
                    logger.warning("Download failed for %s - %s", slug, song.title)

        all_songs.extend(accepted)

    # Save combined song index
    index_path = output_dir / "song_index.jsonl"
    save_song_index(all_songs, index_path)

    # Build audio manifest
    manifest_path = output_dir / "audio_manifest.jsonl"
    build_audio_manifest(all_songs, audio_dir, manifest_path)

    # Check balance
    ok = check_song_balance(index_path, min_songs=10)
    if not ok:
        logger.warning("Some producers have fewer than 10 accepted songs!")


if __name__ == "__main__":
    main()
