"""Probability calibration and confidence signals for multiclass classifiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return np.log(probabilities)


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float = 1.0

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        true_indices: np.ndarray,
    ) -> "TemperatureScaler":
        return cls.fit_logits(probabilities_to_logits(probabilities), true_indices)

    @classmethod
    def fit_logits(
        cls,
        logits: np.ndarray,
        true_indices: np.ndarray,
    ) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        true_indices = np.asarray(true_indices, dtype=int)

        def negative_log_likelihood(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            calibrated = _softmax(logits / temperature)
            selected = calibrated[np.arange(len(true_indices)), true_indices]
            return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(np.log(0.05), np.log(20.0)),
            method="bounded",
        )
        return cls(temperature=float(np.exp(result.x)))

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self.transform_logits(probabilities_to_logits(probabilities))

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        return _softmax(logits / self.temperature)


def confidence_signals(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    top1 = probabilities[np.arange(len(probabilities)), order[:, 0]]
    top2 = probabilities[np.arange(len(probabilities)), order[:, 1]]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )
    normalized_entropy = entropy / np.log(probabilities.shape[1])
    return {
        "confidence": top1,
        "margin": top1 - top2,
        "entropy": normalized_entropy,
    }


def select_rejection_threshold(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
    target_precision: float = 0.8,
    minimum_coverage: float = 0.1,
) -> dict[str, float]:
    """Choose the highest-coverage confidence threshold meeting target precision."""
    signals = confidence_signals(probabilities)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == np.asarray(true_indices)
    candidates = np.unique(signals["confidence"])
    best = None
    for threshold in candidates:
        accepted = signals["confidence"] >= threshold
        coverage = float(accepted.mean())
        if coverage < minimum_coverage or not accepted.any():
            continue
        precision = float(correct[accepted].mean())
        if precision >= target_precision:
            candidate = {
                "threshold": float(threshold),
                "coverage": coverage,
                "precision": precision,
            }
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate

    if best is None:
        threshold = float(np.quantile(signals["confidence"], 1.0 - minimum_coverage))
        accepted = signals["confidence"] >= threshold
        best = {
            "threshold": threshold,
            "coverage": float(accepted.mean()),
            "precision": float(correct[accepted].mean()) if accepted.any() else 0.0,
        }
    return best
