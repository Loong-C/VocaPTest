"""Evaluation utilities for retrieval and classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def top_k_accuracy(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k_values: list[int] = [1, 3, 5],
) -> dict[int, float]:
    """Compute Top-K accuracy.

    Args:
        y_true: (n,) true labels (integer encoded).
        y_pred_probs: (n, num_classes) predicted probabilities.
        k_values: List of k values.

    Returns:
        Dict mapping k to accuracy.
    """
    results = {}
    for k in k_values:
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
        results[k] = float(np.mean(correct))
    return results


def mean_reciprocal_rank(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
) -> float:
    """Compute Mean Reciprocal Rank."""
    ranks = np.argsort(np.argsort(-y_pred_probs, axis=1), axis=1)
    rr = 1.0 / (ranks[np.arange(len(y_true)), y_true] + 1)
    return float(np.mean(rr))


def evaluate_retrieval(
    retrieval_results: list[list[dict]],
    ground_truth: list[str],
) -> dict:
    """Evaluate retrieval results.

    Args:
        retrieval_results: List of result lists, each sorted by score descending.
        ground_truth: True producer slugs.

    Returns:
        Dict with metrics.
    """
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = len(ground_truth)

    for results, gt in zip(retrieval_results, ground_truth):
        predicted_slugs = [r["producer_slug"] for r in results]
        if gt in predicted_slugs[:1]:
            top1_correct += 1
        if gt in predicted_slugs[:3]:
            top3_correct += 1
        if gt in predicted_slugs[:5]:
            top5_correct += 1

    metrics = {
        "total": total,
        "top1_accuracy": top1_correct / total if total > 0 else 0,
        "top3_accuracy": top3_correct / total if total > 0 else 0,
        "top5_accuracy": top5_correct / total if total > 0 else 0,
    }

    logger.info("Retrieval evaluation: %s", metrics)
    return metrics
