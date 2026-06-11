#!/usr/bin/env python
"""Split embedding manifest into train/test by song (stratified per producer)."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Train/test split by song")
    parser.add_argument("--embeddings", required=True, help="Path to embedding manifest JSONL")
    parser.add_argument("--test-songs", type=int, default=2, help="Songs per producer to hold out for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-out", required=True, help="Output path for train manifest")
    parser.add_argument("--test-out", required=True, help="Output path for test manifest")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all records
    records = []
    with open(args.embeddings, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Group segments by (producer, song_id)
    song_segments = defaultdict(list)
    for r in records:
        key = (r["producer_slug"], r["song_id"])
        song_segments[key].append(r)

    # Group songs by producer
    producer_songs = defaultdict(list)
    for (producer, song_id), segs in song_segments.items():
        producer_songs[producer].append((song_id, segs))

    # Split: hold out test_songs per producer
    test_records = []
    train_records = []
    test_song_ids = set()

    for producer, songs in sorted(producer_songs.items()):
        random.shuffle(songs)
        n_test = min(args.test_songs, len(songs) - 1)  # ensure at least 1 train
        n_test = max(1, n_test)  # at least 1 test

        test_songs = songs[:n_test]
        train_songs = songs[n_test:]

        logger.info("%s: %d train + %d test songs (%d + %d segments)",
                     producer, len(train_songs), len(test_songs),
                     sum(len(s[1]) for s in train_songs),
                     sum(len(s[1]) for s in test_songs))

        for sid, segs in train_songs:
            train_records.extend(segs)
        for sid, segs in test_songs:
            test_records.extend(segs)
            test_song_ids.add(sid)

    # Write outputs
    for path, recs in [(args.train_out, train_records), (args.test_out, test_records)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_songs = sum(len(v) for v in producer_songs.values())
    logger.info("Split complete: %d train segments, %d test segments (%d test songs from %d total)",
                 len(train_records), len(test_records), len(test_song_ids), total_songs)

    # Print test song list
    print(f"\nTest songs ({len(test_song_ids)}):")
    for (producer, song_id) in sorted(song_segments):
        if song_id in test_song_ids:
            print(f"  {producer:20s} | {song_id}")


if __name__ == "__main__":
    main()
