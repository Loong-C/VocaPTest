"""Nonnegative probability fusion of per-layer Shrinkage LDA models."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from vocaptest.data.metadata_schema import SearchResult
from vocaptest.features.layer_features import pool_segment_layers
from vocaptest.models.calibration import TemperatureScaler, confidence_signals


def _multiclass_decision_scores(estimator, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(estimator.decision_function(features))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def optimize_nonnegative_weights(
    layer_probabilities: np.ndarray,
    true_indices: np.ndarray,
) -> np.ndarray:
    """Minimize multiclass log loss with nonnegative weights summing to one."""
    layer_probabilities = np.asarray(layer_probabilities, dtype=np.float64)
    true_indices = np.asarray(true_indices, dtype=int)
    n_layers = layer_probabilities.shape[0]

    def objective(weights: np.ndarray) -> float:
        fused = np.tensordot(weights, layer_probabilities, axes=(0, 0))
        selected = fused[np.arange(len(true_indices)), true_indices]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize(
        objective,
        x0=np.full(n_layers, 1.0 / n_layers),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_layers,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Layer weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


@dataclass(frozen=True)
class LayerFusionPrediction:
    results: list[SearchResult]
    accepted: bool
    confidence: float
    margin: float
    entropy: float


class LayerFusionLDA:
    format_version = 3

    def __init__(
        self,
        estimators: list,
        layer_weights: np.ndarray,
        temperature_scaler: TemperatureScaler,
        rejection_threshold: float,
        layer_indices: list[int] | None = None,
        calibration_input: str = "probabilities",
        display_names: dict[str, str] | None = None,
        catalog: dict | None = None,
        embedding_backend: str = "mert_95_all_layers",
    ):
        self.estimators = estimators
        self.layer_weights = np.asarray(layer_weights, dtype=np.float64)
        self.temperature_scaler = temperature_scaler
        self.rejection_threshold = float(rejection_threshold)
        self.layer_indices = layer_indices or list(range(len(estimators)))
        self.calibration_input = calibration_input
        self.display_names = display_names or {}
        self.catalog = catalog or {}
        self.embedding_backend = embedding_backend
        self.classes_ = np.asarray(estimators[0].classes_)
        if len(estimators) != len(self.layer_weights):
            raise ValueError("Each layer estimator needs exactly one fusion weight")
        if len(estimators) != len(self.layer_indices):
            raise ValueError("Each layer estimator needs exactly one layer index")

    def predict_proba_from_layer_features(
        self,
        layer_features: np.ndarray,
    ) -> np.ndarray:
        if self.calibration_input == "logits":
            scores = np.stack([
                _multiclass_decision_scores(
                    estimator,
                    layer_features[:, source_layer, :],
                )
                for source_layer, estimator in zip(self.layer_indices, self.estimators)
            ])
            fused_scores = np.tensordot(self.layer_weights, scores, axes=(0, 0))
            return self.temperature_scaler.transform_logits(fused_scores)
        if self.calibration_input != "probabilities":
            raise ValueError(f"Unknown calibration input: {self.calibration_input}")
        probabilities = np.stack([
            estimator.predict_proba(layer_features[:, source_layer, :])
            for source_layer, estimator in zip(self.layer_indices, self.estimators)
        ])
        fused = np.tensordot(self.layer_weights, probabilities, axes=(0, 0))
        return self.temperature_scaler.transform(fused)

    def rank_segment_layers(
        self,
        segment_layer_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> LayerFusionPrediction:
        layer_features = pool_segment_layers(
            segment_layer_embeddings,
            mode="mean",
        )[None, :, :]
        probabilities = self.predict_proba_from_layer_features(layer_features)[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        signals = confidence_signals(probabilities[None, :])
        confidence = float(signals["confidence"][0])
        return LayerFusionPrediction(
            results=[
                SearchResult(
                    producer_slug=str(self.classes_[index]),
                    display_name=self.display_names.get(
                        str(self.classes_[index]), str(self.classes_[index])
                    ),
                    score=float(probabilities[index]),
                    rank=rank,
                )
                for rank, index in enumerate(order, start=1)
            ],
            accepted=confidence >= self.rejection_threshold,
            confidence=confidence,
            margin=float(signals["margin"][0]),
            entropy=float(signals["entropy"][0]),
        )

    def to_reference_library(self) -> dict:
        song_counts = self.catalog.get("song_counts", {})
        segment_counts = self.catalog.get("segment_counts", {})
        if len(self.layer_indices) == 1:
            backend = (
                f"mert_95+layer_{self.layer_indices[0]}_song_mean_"
                "shrinkage_lda_calibrated"
            )
        else:
            backend = "mert_95+nonnegative_layer_fusion_lda_calibrated"
        return {
            "backend": backend,
            "producers": {
                str(slug): {
                    "display_name": self.display_names.get(str(slug), str(slug)),
                    "song_count": song_counts.get(str(slug)),
                    "segment_count": segment_counts.get(str(slug)),
                }
                for slug in self.classes_
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({
                "format_version": self.format_version,
                "estimators": self.estimators,
                "layer_weights": self.layer_weights,
                "temperature": self.temperature_scaler.temperature,
                "rejection_threshold": self.rejection_threshold,
                "layer_indices": self.layer_indices,
                "calibration_input": self.calibration_input,
                "display_names": self.display_names,
                "catalog": self.catalog,
                "embedding_backend": self.embedding_backend,
            }, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LayerFusionLDA":
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        if artifact.get("format_version") != cls.format_version:
            raise ValueError("Unsupported LayerFusionLDA artifact version")
        return cls(
            estimators=artifact["estimators"],
            layer_weights=artifact["layer_weights"],
            temperature_scaler=TemperatureScaler(artifact["temperature"]),
            rejection_threshold=artifact["rejection_threshold"],
            layer_indices=artifact.get("layer_indices"),
            calibration_input=artifact.get("calibration_input", "probabilities"),
            display_names=artifact.get("display_names"),
            catalog=artifact.get("catalog"),
            embedding_backend=artifact.get("embedding_backend", "mert_95_all_layers"),
        )
