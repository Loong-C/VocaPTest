"""Build one normalized feature vector per song from segment embeddings."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.features.extract_embeddings import load_all_embeddings_aligned


@dataclass(frozen=True)
class SongFeatureMetadata:
    song_id: str
    work_id: str
    producer_slug: str
    title: str
    segment_count: int


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def mean_segment_embeddings(segment_embeddings: np.ndarray) -> np.ndarray:
    """Average segment embeddings and normalize the resulting song vector."""
    if segment_embeddings.ndim != 2 or len(segment_embeddings) == 0:
        raise ValueError("Expected a non-empty 2D segment embedding matrix")
    return l2_normalize(segment_embeddings.mean(axis=0))


def build_song_feature_matrix(
    records: list[EmbeddingRecord],
) -> tuple[np.ndarray, list[SongFeatureMetadata]]:
    embeddings, loaded_records = load_all_embeddings_aligned(records)
    if not loaded_records:
        raise ValueError("No readable embeddings were supplied")

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(loaded_records):
        grouped_indices[record.song_id].append(index)

    song_vectors: list[np.ndarray] = []
    metadata: list[SongFeatureMetadata] = []
    for song_id in sorted(grouped_indices):
        indices = grouped_indices[song_id]
        song_records = [loaded_records[index] for index in indices]
        producer_slug = song_records[0].producer_slug
        if any(record.producer_slug != producer_slug for record in song_records):
            raise ValueError(f"Song {song_id} has conflicting producer labels")

        song_vectors.append(mean_segment_embeddings(embeddings[indices]))
        metadata.append(SongFeatureMetadata(
            song_id=song_id,
            work_id=song_records[0].work_id or song_id,
            producer_slug=producer_slug,
            title=song_records[0].title or song_id,
            segment_count=len(indices),
        ))

    return np.stack(song_vectors).astype(np.float32), metadata


def save_song_features(
    features: np.ndarray,
    metadata: list[SongFeatureMetadata],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "song_features.npy", features)
    with open(output_dir / "songs.jsonl", "w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
