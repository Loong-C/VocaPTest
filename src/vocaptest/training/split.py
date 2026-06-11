"""Train/val/test split utilities — ensures song-level splitting."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def split_by_song(
    records: list[EmbeddingRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split song IDs into train/val/test sets.

    Returns:
        (train_song_ids, val_song_ids, test_song_ids)
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be valid fractions")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    song_to_producer: dict[str, str] = {}
    producer_to_songs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        previous = song_to_producer.setdefault(record.song_id, record.producer_slug)
        if previous != record.producer_slug:
            raise ValueError(f"Song {record.song_id} has conflicting producer labels")
        producer_to_songs[record.producer_slug].add(record.song_id)

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    test_ratio = 1.0 - train_ratio - val_ratio

    for producer_slug in sorted(producer_to_songs):
        songs = np.array(sorted(producer_to_songs[producer_slug]), dtype=object)
        songs = songs[rng.permutation(len(songs))]
        n_songs = len(songs)
        if n_songs < 3 and val_ratio > 0 and test_ratio > 0:
            raise ValueError(
                f"Producer {producer_slug} needs at least 3 songs for train/val/test"
            )

        n_val = max(1, int(round(n_songs * val_ratio))) if val_ratio > 0 else 0
        n_test = max(1, int(round(n_songs * test_ratio))) if test_ratio > 0 else 0
        while n_val + n_test >= n_songs:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(
                    f"Producer {producer_slug} has too few songs for requested ratios"
                )

        n_train = n_songs - n_val - n_test
        train_ids.extend(songs[:n_train].tolist())
        val_ids.extend(songs[n_train:n_train + n_val].tolist())
        test_ids.extend(songs[n_train + n_val:].tolist())

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()
    n = len(song_to_producer)

    logger.info(
        "Split: %d songs -> train=%d, val=%d, test=%d",
        n, len(train_ids), len(val_ids), len(test_ids),
    )
    return train_ids, val_ids, test_ids


def save_splits(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    output_dir: str | Path,
) -> None:
    """Save split files to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = output_dir / f"{name}_song_ids.txt"
        with open(path, "w", encoding="utf-8") as f:
            for song_id in sorted(ids):
                f.write(song_id + "\n")
    logger.info("Saved splits to %s", output_dir)
