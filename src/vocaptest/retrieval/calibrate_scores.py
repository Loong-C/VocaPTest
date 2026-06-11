"""Score calibration utilities for improving result interpretability."""
from __future__ import annotations

import numpy as np

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def softmax_scores(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Apply softmax to raw similarity scores for a probability-like output.

    Args:
        scores: Raw similarity scores.
        temperature: Softmax temperature (lower = sharper distribution).

    Returns:
        Softmax-normalized scores that sum to 1.
    """
    arr = np.array(scores) / temperature
    arr = arr - arr.max()  # numerical stability
    exp = np.exp(arr)
    return list(exp / exp.sum())


def calibrate_thresholds(
    all_scores: list[list[float]],
    labels: list[str],
    target_producer: str,
    target_recall: float = 0.9,
) -> float:
    """Find a score threshold that achieves target recall for a producer.

    Args:
        all_scores: List of score lists from evaluation.
        labels: Ground-truth producer slugs.
        target_producer: The producer to calibrate for.
        target_recall: Desired recall (0-1).

    Returns:
        Score threshold.
    """
    scores_for_producer = []
    for scores, label in zip(all_scores, labels):
        if label == target_producer:
            # Find the score assigned to this producer
            for s in scores:
                if isinstance(s, dict):
                    if s.get("producer_slug") == target_producer:
                        scores_for_producer.append(s["score"])
                        break
                elif isinstance(s, (int, float)):
                    scores_for_producer.append(s)

    if not scores_for_producer:
        return 0.0

    scores_for_producer.sort()
    idx = int((1 - target_recall) * len(scores_for_producer))
    threshold = scores_for_producer[max(0, min(idx, len(scores_for_producer) - 1))]
    logger.info(
        "Calibrated threshold for %s: %.4f (recall=%.2f)",
        target_producer, threshold, target_recall,
    )
    return threshold
