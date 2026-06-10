"""Pooling strategies for aggregating segment embeddings into song embeddings."""
from __future__ import annotations

import numpy as np


def mean_pooling(embeddings: np.ndarray) -> np.ndarray:
    """Mean pooling over segments.

    Args:
        embeddings: (num_segments, dim) array.

    Returns:
        (dim,) array.
    """
    return embeddings.mean(axis=0)


def max_pooling(embeddings: np.ndarray) -> np.ndarray:
    """Max pooling over segments."""
    return embeddings.max(axis=0)


def attention_pooling(
    embeddings: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted pooling.

    Args:
        embeddings: (num_segments, dim).
        weights: (num_segments,) optional weights. If None, uses uniform.
    """
    if weights is None:
        weights = np.ones(embeddings.shape[0]) / embeddings.shape[0]
    weights = weights / weights.sum()
    return (embeddings * weights[:, np.newaxis]).sum(axis=0)


def top_k_pooling(
    embeddings: np.ndarray,
    reference: np.ndarray,
    top_ratio: float = 0.4,
) -> np.ndarray:
    """Pool top-k segments by similarity to a reference vector.

    Args:
        embeddings: (num_segments, dim).
        reference: (dim,) reference vector (e.g., producer centroid).
        top_ratio: Fraction of top segments to keep.

    Returns:
        (dim,) array.
    """
    sims = np.dot(embeddings, reference)
    k = max(1, int(len(sims) * top_ratio))
    top_idx = np.argsort(sims)[-k:]
    return embeddings[top_idx].mean(axis=0)


POOLING_MAP = {
    "mean": mean_pooling,
    "max": max_pooling,
    "attention": attention_pooling,
}


def get_pooling(name: str):
    """Get a pooling function by name."""
    fn = POOLING_MAP.get(name)
    if fn is None:
        raise ValueError(f"Unknown pooling: {name}. Available: {list(POOLING_MAP)}")
    return fn
