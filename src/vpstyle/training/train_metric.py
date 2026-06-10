"""Metric learning stub — placeholder for future contrastive/triplet training."""
from __future__ import annotations

from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def train_metric_learning(*args, **kwargs):
    """Placeholder for metric learning (e.g., triplet loss, contrastive loss).

    Not implemented in MVP. Will be used to fine-tune MERT/MuQ embeddings
    for producer-specific similarity.
    """
    logger.warning("Metric learning is not implemented in MVP.")
    raise NotImplementedError("Metric learning not implemented in MVP")
