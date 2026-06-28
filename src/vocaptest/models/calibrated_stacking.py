"""Calibrated stacking classifier for frozen MERT layer features."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vocaptest.data.metadata_schema import SearchResult
from vocaptest.features.layer_features import pool_segment_layers
from vocaptest.models.calibration import confidence_signals


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _align_probabilities(
    probabilities: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(target_classes)), dtype=np.float64)
    target_lookup = {str(label): index for index, label in enumerate(target_classes)}
    for source_index, label in enumerate(source_classes):
        aligned[:, target_lookup[str(label)]] = probabilities[:, source_index]
    return _normalize(aligned)


def _flatten_layers(layer_features: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    return layer_features[:, list(layers), :].reshape(layer_features.shape[0], -1)


def _meta_features(probabilities: np.ndarray, mode: str) -> np.ndarray:
    if mode == "prob":
        return probabilities.reshape(probabilities.shape[0], -1)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    if mode == "log_prob":
        return log_probabilities.reshape(probabilities.shape[0], -1)
    if mode == "prob_and_log":
        return np.concatenate(
            [
                probabilities.reshape(probabilities.shape[0], -1),
                log_probabilities.reshape(probabilities.shape[0], -1),
            ],
            axis=1,
        )
    raise ValueError(f"Unknown stacking meta-feature mode: {mode}")


@dataclass(frozen=True)
class CalibratedStackingPrediction:
    results: list[SearchResult]
    accepted: bool
    confidence: float
    margin: float
    entropy: float


class CalibratedStackingLDA:
    """Stack several global LDA heads with a calibrated meta classifier."""

    format_version = 1

    def __init__(
        self,
        base_heads: list[dict],
        meta_model,
        meta_scaler,
        rejection_threshold: float,
        meta_feature_mode: str = "log_prob",
        display_names: dict[str, str] | None = None,
        catalog: dict | None = None,
        embedding_backend: str = "mert_95_p4_calibrated_stacking",
    ):
        if not base_heads:
            raise ValueError("CalibratedStackingLDA requires at least one base head")
        self.base_heads = base_heads
        self.meta_model = meta_model
        self.meta_scaler = meta_scaler
        self.rejection_threshold = float(rejection_threshold)
        self.meta_feature_mode = meta_feature_mode
        self.display_names = display_names or {}
        self.catalog = catalog or {}
        self.embedding_backend = embedding_backend
        self.classes_ = np.asarray(meta_model.classes_)

    def _predict_base_head(self, head: dict, layer_features: np.ndarray) -> np.ndarray:
        kind = head["kind"]
        layers = tuple(head["layers"])
        if kind == "layer":
            estimator = head["estimator"]
            probabilities = estimator.predict_proba(layer_features[:, layers[0], :])
            return _align_probabilities(probabilities, estimator.classes_, self.classes_)
        if kind == "layer_fusion":
            outputs = []
            for layer, estimator in zip(layers, head["estimators"]):
                probabilities = estimator.predict_proba(layer_features[:, layer, :])
                outputs.append(
                    _align_probabilities(probabilities, estimator.classes_, self.classes_)
                )
            return _normalize(np.mean(np.stack(outputs), axis=0))
        if kind == "concat":
            estimator = head["estimator"]
            probabilities = estimator.predict_proba(_flatten_layers(layer_features, layers))
            return _align_probabilities(probabilities, estimator.classes_, self.classes_)
        raise ValueError(f"Unknown base head kind: {kind}")

    def predict_base_probabilities(self, layer_features: np.ndarray) -> np.ndarray:
        """Return probabilities with shape (songs, base_heads, classes)."""
        return np.stack(
            [self._predict_base_head(head, layer_features) for head in self.base_heads],
            axis=1,
        )

    def predict_proba_from_layer_features(self, layer_features: np.ndarray) -> np.ndarray:
        base_probabilities = self.predict_base_probabilities(layer_features)
        features = _meta_features(base_probabilities, self.meta_feature_mode)
        features = self.meta_scaler.transform(features)
        probabilities = self.meta_model.predict_proba(features)
        return _align_probabilities(probabilities, self.meta_model.classes_, self.classes_)

    def rank_segment_layers(
        self,
        segment_layer_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> CalibratedStackingPrediction:
        layer_features = pool_segment_layers(
            segment_layer_embeddings,
            mode="mean",
        )[None, :, :]
        probabilities = self.predict_proba_from_layer_features(layer_features)[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        signals = confidence_signals(probabilities[None, :])
        confidence = float(signals["confidence"][0])
        return CalibratedStackingPrediction(
            results=[
                SearchResult(
                    producer_slug=str(self.classes_[index]),
                    display_name=self.display_names.get(
                        str(self.classes_[index]),
                        str(self.classes_[index]),
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
        return {
            "backend": self.embedding_backend,
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
            pickle.dump(
                {
                    "format_version": self.format_version,
                    "base_heads": self.base_heads,
                    "meta_model": self.meta_model,
                    "meta_scaler": self.meta_scaler,
                    "rejection_threshold": self.rejection_threshold,
                    "meta_feature_mode": self.meta_feature_mode,
                    "display_names": self.display_names,
                    "catalog": self.catalog,
                    "embedding_backend": self.embedding_backend,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedStackingLDA":
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        if artifact.get("format_version") != cls.format_version:
            raise ValueError("Unsupported CalibratedStackingLDA artifact version")
        return cls(
            base_heads=artifact["base_heads"],
            meta_model=artifact["meta_model"],
            meta_scaler=artifact["meta_scaler"],
            rejection_threshold=artifact["rejection_threshold"],
            meta_feature_mode=artifact.get("meta_feature_mode", "log_prob"),
            display_names=artifact.get("display_names"),
            catalog=artifact.get("catalog"),
            embedding_backend=artifact.get(
                "embedding_backend",
                "mert_95_p4_calibrated_stacking",
            ),
        )
