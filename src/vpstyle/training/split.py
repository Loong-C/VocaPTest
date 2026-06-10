"""Train/val/test split utilities — ensures song-level splitting."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from vpstyle.data.metadata_schema import EmbeddingRecord
from vpstyle.utils.logging import setup_logging

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
    # Group records by song_id
    song_ids = list(set(r.song_id for r in records))
    random.seed(seed)
    random.shuffle(song_ids)

    n = len(song_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = song_ids[:n_train]
    val_ids = song_ids[n_train:n_train + n_val]
    test_ids = song_ids[n_train + n_val:]

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
