#!/usr/bin/env python
"""Generate song name mapping from youtube_songs.jsonl and embedding manifest."""
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate song name mapping")
    parser.add_argument("--songs", required=True, help="Path to youtube_songs.jsonl")
    parser.add_argument("--embeddings", required=True, help="Path to embedding manifest JSONL (for segment counts)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Load song metadata
    song_map = {}
    with open(args.songs, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "accepted":
                continue
            song_map[rec["song_id"]] = {
                "title": rec["title"],
                "producer_slug": rec["producer_slug"],
            }

    # Count segments per song from embedding manifest
    segment_counts = defaultdict(int)
    with open(args.embeddings, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            segment_counts[rec["song_id"]] += 1

    # Write CSV
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["producer_slug", "song_id", "title", "segments"])
        for sid in sorted(song_map.keys()):
            info = song_map[sid]
            writer.writerow([
                info["producer_slug"],
                sid,
                info["title"],
                segment_counts.get(sid, 0),
            ])

    print(f"Saved {len(song_map)} songs to {args.output}")


if __name__ == "__main__":
    main()
