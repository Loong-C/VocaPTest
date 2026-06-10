#!/usr/bin/env python
"""Fetch VocaDB metadata for all producers in config."""
import argparse
from pathlib import Path

import yaml

from vpstyle.data.vocadb_client import VocaDBClient
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Fetch VocaDB metadata")
    parser.add_argument("--producers", required=True, help="Path to producers.yaml")
    parser.add_argument("--output", required=True, help="Output directory for raw JSONL")
    parser.add_argument("--max-songs", type=int, default=200, help="Max songs per producer")
    args = parser.parse_args()

    with open(args.producers, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = VocaDBClient()

    for producer in data["producers"]:
        slug = producer["slug"]
        artist_id = producer.get("vocadb_artist_id")

        if artist_id is None:
            # Search for artist
            display_name = producer["display_name"]
            logger.info("Searching for artist: %s", display_name)
            results = client.search_artists(display_name, max_results=5)
            if not results:
                logger.warning("No artist found for %s — skipping", display_name)
                continue

            # Print candidates for manual review
            print(f"\n--- Candidates for {display_name} ---")
            for i, artist in enumerate(results):
                print(f"  [{i}] id={artist['id']} name={artist.get('name', '?')} "
                      f"({artist.get('artistType', '?')})")
            print("  Enter index to select, or 's' to skip: ", end="")
            choice = input().strip()
            if choice.lower() == "s":
                continue
            try:
                artist_id = results[int(choice)]["id"]
            except (ValueError, IndexError):
                logger.warning("Invalid choice — skipping %s", display_name)
                continue

        logger.info("Fetching songs for %s (artist_id=%d)", slug, artist_id)
        songs = client.get_songs_by_artist(
            artist_id,
            fields="PVs,Artists,Tags",
            max_results=args.max_songs,
        )

        out_path = output_dir / f"{slug}_songs.jsonl"
        client.save_raw_json(songs, out_path)
        logger.info("Saved %d songs for %s", len(songs), slug)


if __name__ == "__main__":
    main()
