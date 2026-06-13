"""Song-level pooling for cached per-segment, per-layer embeddings."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.features.extract_embeddings import resolve_embedding_path
from vocaptest.features.song_features import SongFeatureMetadata, l2_normalize


def pool_segment_layers(
    segment_layers: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """Pool `(segments, layers, dim)` into `(layers, feature_dim)`."""
    if segment_layers.ndim != 3 or len(segment_layers) == 0:
        raise ValueError("Expected a non-empty (segments, layers, dim) array")

    mean = segment_layers.mean(axis=0)
    if mode == "mean":
        pooled = mean
    elif mode == "mean_std":
        pooled = np.concatenate([mean, segment_layers.std(axis=0)], axis=1)
    elif mode == "mean_std_change":
        if len(segment_layers) > 1:
            change = np.abs(np.diff(segment_layers, axis=0)).mean(axis=0)
        else:
            change = np.zeros_like(mean)
        pooled = np.concatenate([mean, segment_layers.std(axis=0), change], axis=1)
    elif mode == "multi_stat":
        if len(segment_layers) > 1:
            delta = segment_layers[-1] - segment_layers[0]
            change = np.abs(np.diff(segment_layers, axis=0)).mean(axis=0)
        else:
            delta = np.zeros_like(mean)
            change = np.zeros_like(mean)
        pooled = np.concatenate([
            mean,
            segment_layers.std(axis=0),
            np.quantile(segment_layers, 0.25, axis=0),
            np.quantile(segment_layers, 0.75, axis=0),
            delta,
            change,
        ], axis=1)
    else:
        raise ValueError(f"Unknown song pooling mode: {mode}")

    return np.stack([l2_normalize(layer) for layer in pooled]).astype(np.float32)


def build_song_layer_feature_matrix(
    records: list[EmbeddingRecord],
    mode: str = "mean",
) -> tuple[np.ndarray, list[SongFeatureMetadata]]:
    by_song: dict[str, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        by_song[record.song_id].append(record)

    song_features: list[np.ndarray] = []
    metadata: list[SongFeatureMetadata] = []
    for song_id in tqdm(sorted(by_song), desc=f"Pooling songs ({mode})"):
        song_records = sorted(by_song[song_id], key=lambda item: item.segment_id)
        arrays = [
            np.load(resolve_embedding_path(record))
            for record in song_records
        ]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(f"Inconsistent layer embedding shapes for {song_id}: {shapes}")
        producer_slug = song_records[0].producer_slug
        if any(record.producer_slug != producer_slug for record in song_records):
            raise ValueError(f"Song {song_id} has conflicting producer labels")

        song_features.append(pool_segment_layers(np.stack(arrays), mode=mode))
        metadata.append(SongFeatureMetadata(
            song_id=song_id,
            work_id=song_records[0].work_id or song_id,
            producer_slug=producer_slug,
            title=song_records[0].title or song_id,
            segment_count=len(song_records),
        ))

    return np.stack(song_features), metadata


def build_song_segment_feature_tensor(
    records: list[EmbeddingRecord],
    layer: int,
) -> tuple[np.ndarray, np.ndarray, list[SongFeatureMetadata]]:
    """Load one MERT layer into a padded `(songs, segments, dim)` tensor."""
    by_song: dict[str, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        by_song[record.song_id].append(record)
    if not by_song:
        raise ValueError("Cannot build segment tensor from an empty manifest")

    songs: list[np.ndarray] = []
    metadata: list[SongFeatureMetadata] = []
    for song_id in tqdm(sorted(by_song), desc=f"Loading layer {layer} segments"):
        song_records = sorted(by_song[song_id], key=lambda item: item.segment_id)
        arrays = [
            np.load(resolve_embedding_path(record))
            for record in song_records
        ]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(f"Inconsistent layer embedding shapes for {song_id}: {shapes}")
        if not 0 <= layer < arrays[0].shape[0]:
            raise ValueError(
                f"Layer {layer} is unavailable for {song_id}: {arrays[0].shape}"
            )
        producer_slug = song_records[0].producer_slug
        if any(record.producer_slug != producer_slug for record in song_records):
            raise ValueError(f"Song {song_id} has conflicting producer labels")
        songs.append(np.stack([array[layer] for array in arrays]))
        metadata.append(SongFeatureMetadata(
            song_id=song_id,
            work_id=song_records[0].work_id or song_id,
            producer_slug=producer_slug,
            title=song_records[0].title or song_id,
            segment_count=len(song_records),
        ))

    max_segments = max(len(song) for song in songs)
    feature_dim = songs[0].shape[1]
    features = np.zeros(
        (len(songs), max_segments, feature_dim),
        dtype=np.float32,
    )
    masks = np.zeros((len(songs), max_segments), dtype=bool)
    for index, song in enumerate(songs):
        features[index, :len(song)] = song
        masks[index, :len(song)] = True
    return features, masks, metadata
