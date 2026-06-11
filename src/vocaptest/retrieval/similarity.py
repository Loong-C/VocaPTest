"""Similarity computation utilities."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of vectors.

    Args:
        a: (m, d) or (d,) array.
        b: (n, d) or (d,) array.

    Returns:
        (m, n) similarity matrix.
    """
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    a_norm = normalize(a.astype(np.float64), norm="l2")
    b_norm = normalize(b.astype(np.float64), norm="l2")
    return np.dot(a_norm, b_norm.T)


def score_song_against_producer(
    song_segment_embs: np.ndarray,
    producer_centroids: np.ndarray,
    top_ratio: float = 0.4,
) -> float:
    """Score a song against a producer using segment-level top-k aggregation.

    For each segment, find the best-matching centroid. Then take the mean of
    the top (top_ratio) segment scores.

    Args:
        song_segment_embs: (num_segments, dim) array of segment embeddings.
        producer_centroids: (num_centroids, dim) array of producer centroids.
        top_ratio: Fraction of top segment scores to average.

    Returns:
        Float cosine score in [-1, 1]. This is a similarity, not a probability.
    """
    sims = cosine_similarity(song_segment_embs, producer_centroids)
    per_segment_best = sims.max(axis=1)  # best centroid per segment
    k = max(1, int(len(per_segment_best) * top_ratio))
    top_scores = np.sort(per_segment_best)[-k:]
    return float(np.mean(top_scores))


def score_song_against_all(
    song_segment_embs: np.ndarray,
    profiles: dict,
    top_ratio: float = 0.4,
) -> list[dict]:
    """Score a song against all producer profiles.

    Args:
        song_segment_embs: (num_segments, dim) embeddings.
        profiles: Producer profiles dict from build_profiles.
        top_ratio: Top fraction for segment aggregation.

    Returns:
        List of {"producer_slug", "score"} sorted descending.
    """
    results = []
    for slug, profile in profiles["producers"].items():
        score = score_song_against_producer(
            song_segment_embs,
            profile["centroids"],
            top_ratio=top_ratio,
        )
        results.append({
            "producer_slug": slug,
            "display_name": profile.get("display_name", slug),
            "score": round(score, 4),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
