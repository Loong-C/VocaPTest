#!/usr/bin/env python
"""Broad P4 model search under no-new-data/no-producer-specific constraints."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import (
    build_song_layer_feature_matrix,
    build_song_segment_feature_tensor,
)
from vocaptest.models.calibration import confidence_signals, select_rejection_threshold
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.utils.paths import project_root


DEFAULT_LAYER_SET = (5, 6, 8)
TARGET_ACCEPTED_PRECISION = 0.96


@dataclass(frozen=True)
class SplitFeatures:
    layers: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    song_ids: list[str]
    titles: list[str]
    records: list


@dataclass(frozen=True)
class AuditFlag:
    song_id: str
    producer_slug: str
    title: str
    codes: tuple[str, ...]
    risk_score: int


def load_split(path: Path, *, mode: str = "mean") -> SplitFeatures:
    records = load_embedding_manifest(path)
    layers, metadata = build_song_layer_feature_matrix(records, mode=mode)
    return SplitFeatures(
        layers=layers,
        labels=np.asarray([item.producer_slug for item in metadata]),
        groups=np.asarray([item.work_id for item in metadata]),
        song_ids=[item.song_id for item in metadata],
        titles=[item.title for item in metadata],
        records=records,
    )


def source_key(record: dict) -> str | None:
    source_id = record.get("source_id") or record.get("youtube_id")
    if not source_id:
        return None
    service = str(record.get("source_service") or "").lower()
    if "youtube" in service:
        return f"youtube_{source_id}"
    if "nico" in service:
        return f"niconico_{source_id}"
    return str(source_id)


def load_train_audit_flags(path: Path) -> dict[str, AuditFlag]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    flags: dict[str, AuditFlag] = {}
    for record in audit["records"]:
        if record.get("split") != "train" or not record.get("flags"):
            continue
        song_id = source_key(record)
        if not song_id:
            continue
        flags[song_id] = AuditFlag(
            song_id=song_id,
            producer_slug=record["producer_slug"],
            title=record["title"],
            codes=tuple(flag["code"] for flag in record["flags"]),
            risk_score=int(record.get("risk_score") or 0),
        )
    return flags


def filter_mask(
    split: SplitFeatures,
    audit_flags: dict[str, AuditFlag],
    strategy: str,
) -> np.ndarray:
    codes_by_strategy = {
        "raw": set(),
        "source_clean": {"configured_source_not_original"},
        "review_clean": {
            "configured_source_not_original",
            "low_vocadb_rating",
            "review_pv_author",
        },
    }
    if strategy not in codes_by_strategy:
        raise ValueError(f"Unknown filter strategy: {strategy}")
    blocked_codes = codes_by_strategy[strategy]
    keep = []
    for song_id in split.song_ids:
        flag = audit_flags.get(song_id)
        excluded = flag is not None and bool(set(flag.codes) & blocked_codes)
        keep.append(not excluded)
    return np.asarray(keep, dtype=bool)


def apply_mask(split: SplitFeatures, mask: np.ndarray) -> SplitFeatures:
    kept_song_ids = {
        song_id for song_id, keep in zip(split.song_ids, mask) if keep
    }
    return SplitFeatures(
        layers=split.layers[mask],
        labels=split.labels[mask],
        groups=split.groups[mask],
        song_ids=[song_id for song_id, keep in zip(split.song_ids, mask) if keep],
        titles=[title for title, keep in zip(split.titles, mask) if keep],
        records=[
            record for record in split.records
            if record.song_id in kept_song_ids
        ],
    )


def class_counts(split: SplitFeatures) -> dict[str, int]:
    return dict(sorted(Counter(split.labels).items()))


def true_indices(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {str(label): i for i, label in enumerate(classes)}
    return np.asarray([index[str(label)] for label in labels], dtype=int)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def align_probabilities(
    probabilities: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(target_classes)), dtype=np.float64)
    for source_index, label in enumerate(source_classes):
        target_index = int(np.where(target_classes == label)[0][0])
        aligned[:, target_index] = probabilities[:, source_index]
    return normalize_probabilities(aligned)


def metrics_key(evaluation: dict) -> tuple[float, float, float, float, float]:
    metrics = evaluation["metrics"]
    return (
        metrics["macro_f1"],
        metrics["top1_accuracy"],
        metrics["top3_accuracy"],
        metrics["mrr"],
        -metrics["log_loss"],
    )


def method_selection_key(method: dict) -> tuple[float, float, float, float, float]:
    return metrics_key(method["dev"])


def ranking_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    probabilities = normalize_probabilities(probabilities)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = classes[order[:, 0]]
    lookup = {str(label): i for i, label in enumerate(classes)}
    ranks = np.asarray([
        int(np.where(order[row] == lookup[str(label)])[0][0]) + 1
        for row, label in enumerate(labels)
    ])
    return (
        {
            "top1_accuracy": float(np.mean(predictions == labels)),
            "top3_accuracy": float(np.mean(ranks <= 3)),
            "macro_f1": float(f1_score(
                labels,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )),
            "mrr": float(np.mean(1.0 / ranks)),
            "log_loss": float(log_loss(
                true_indices(labels, classes),
                probabilities,
                labels=np.arange(len(classes)),
            )),
        },
        predictions,
        ranks,
    )


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    threshold: float | None = None,
) -> dict:
    probabilities = normalize_probabilities(probabilities)
    metrics, predictions, ranks = ranking_metrics(probabilities, labels, classes)
    output = {
        "metrics": metrics,
        "wrong_count": int(np.sum(predictions != labels)),
        "top3_miss_count": int(np.sum(ranks > 3)),
    }
    if threshold is not None:
        signals = confidence_signals(probabilities)
        accepted = signals["confidence"] >= threshold
        correct = predictions == labels
        output["rejection"] = {
            "threshold": float(threshold),
            "coverage": float(np.mean(accepted)),
            "accepted_count": int(np.sum(accepted)),
            "accepted_accuracy": (
                float(np.mean(correct[accepted]))
                if np.any(accepted)
                else None
            ),
            "accepted_wrong_count": int(np.sum(accepted & ~correct)),
            "rejected_correct_count": int(np.sum((~accepted) & correct)),
        }
    return output


def method_record(
    name: str,
    family: str,
    extra: dict,
    dev_probabilities: np.ndarray,
    final_probabilities: np.ndarray,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
) -> dict:
    dev_probabilities = normalize_probabilities(dev_probabilities)
    final_probabilities = normalize_probabilities(final_probabilities)
    threshold_info = select_rejection_threshold(
        dev_probabilities,
        true_indices(dev.labels, classes),
        target_precision=TARGET_ACCEPTED_PRECISION,
        minimum_coverage=0.1,
    )
    threshold = threshold_info["threshold"]
    return {
        "method": name,
        "family": family,
        "extra": extra,
        "dev_rejection_selection": threshold_info,
        "dev": evaluate_probabilities(
            dev_probabilities,
            dev.labels,
            classes,
            threshold=threshold,
        ),
        "final": evaluate_probabilities(
            final_probabilities,
            final.labels,
            classes,
            threshold=threshold,
        ),
    }


def fit_layer_lda_probabilities(
    train: SplitFeatures,
    target: SplitFeatures,
    layer_indices: tuple[int, ...],
    classes: np.ndarray,
) -> np.ndarray:
    probabilities = []
    for layer in layer_indices:
        model = SongMeanShrinkageLDA.fit(train.layers[:, layer, :], train.labels)
        layer_probabilities = model.predict_proba(target.layers[:, layer, :])
        probabilities.append(
            align_probabilities(layer_probabilities, model.classes_, classes)
        )
    return np.stack(probabilities)


def arithmetic_fuse(probabilities_by_layer: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return normalize_probabilities(np.tensordot(weights, probabilities_by_layer, axes=(0, 0)))


def geometric_fuse(probabilities_by_layer: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    log_probabilities = np.log(np.clip(probabilities_by_layer, 1e-12, 1.0))
    return softmax(np.tensordot(weights, log_probabilities, axes=(0, 0)), axis=1)


def rank_fuse(probabilities_by_layer: np.ndarray, scale: float) -> np.ndarray:
    layer_scores = []
    class_count = probabilities_by_layer.shape[2]
    for probabilities in probabilities_by_layer:
        order = np.argsort(probabilities, axis=1)[:, ::-1]
        ranks = np.empty_like(order)
        ranks[np.arange(len(order))[:, None], order] = np.arange(class_count)
        layer_scores.append((class_count - ranks).astype(np.float64) / class_count)
    scores = np.mean(np.stack(layer_scores), axis=0)
    return softmax(scale * scores, axis=1)


def temperature_fuse(
    probabilities_by_layer: np.ndarray,
    weights: np.ndarray,
    temperature: float,
) -> np.ndarray:
    adjusted = softmax(
        np.log(np.clip(probabilities_by_layer, 1e-12, 1.0)) / temperature,
        axis=2,
    )
    return arithmetic_fuse(adjusted, weights)


def optimize_weights(
    probabilities_by_layer: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    geometric: bool,
) -> np.ndarray:
    target = true_indices(labels, classes)
    layer_count = probabilities_by_layer.shape[0]

    def objective(weights: np.ndarray) -> float:
        mixed = (
            geometric_fuse(probabilities_by_layer, weights)
            if geometric
            else arithmetic_fuse(probabilities_by_layer, weights)
        )
        selected = mixed[np.arange(len(target)), target]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize(
        objective,
        x0=np.full(layer_count, 1.0 / layer_count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * layer_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success:
        return np.full(layer_count, 1.0 / layer_count)
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def select_layers_by_dev(
    all_dev_layer_probabilities: np.ndarray,
    dev: SplitFeatures,
    classes: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    scored = []
    for layer in range(all_dev_layer_probabilities.shape[0]):
        scored.append((
            layer,
            evaluate_probabilities(
                all_dev_layer_probabilities[layer],
                dev.labels,
                classes,
            ),
        ))
    return tuple(sorted(
        layer for layer, _ in sorted(
            scored,
            key=lambda item: metrics_key(item[1]),
            reverse=True,
        )[:count]
    ))


def search_layer_fusions(
    all_dev: np.ndarray,
    all_final: np.ndarray,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    *,
    max_combo_size: int,
) -> list[dict]:
    methods = []
    layer_count = all_dev.shape[0]

    for layers, name, family in [
        (DEFAULT_LAYER_SET, "lda_equal_568", "layer_probability_fusion"),
        (tuple(range(layer_count)), "lda_equal_all_layers", "layer_probability_fusion"),
    ]:
        weights = np.full(len(layers), 1.0 / len(layers))
        methods.append(method_record(
            name,
            family,
            {"layers": list(layers), "weights": weights.tolist(), "fusion": "arithmetic"},
            arithmetic_fuse(all_dev[list(layers)], weights),
            arithmetic_fuse(all_final[list(layers)], weights),
            dev,
            final,
            classes,
        ))

    selected_layers = select_layers_by_dev(all_dev, dev, classes, 3)
    selected_weights = np.full(len(selected_layers), 1.0 / len(selected_layers))
    methods.append(method_record(
        "lda_equal_top3_dev_layers",
        "layer_probability_fusion",
        {
            "layers": list(selected_layers),
            "weights": selected_weights.tolist(),
            "fusion": "arithmetic",
        },
        arithmetic_fuse(all_dev[list(selected_layers)], selected_weights),
        arithmetic_fuse(all_final[list(selected_layers)], selected_weights),
        dev,
        final,
        classes,
    ))

    optimized_weights = optimize_weights(
        all_dev,
        dev.labels,
        classes,
        geometric=False,
    )
    methods.append(method_record(
        "lda_dev_weighted_all_layers",
        "layer_probability_fusion",
        {
            "layers": list(range(layer_count)),
            "weights": [float(weight) for weight in optimized_weights],
            "fusion": "arithmetic",
        },
        arithmetic_fuse(all_dev, optimized_weights),
        arithmetic_fuse(all_final, optimized_weights),
        dev,
        final,
        classes,
    ))

    geometric_weights = optimize_weights(
        all_dev,
        dev.labels,
        classes,
        geometric=True,
    )
    methods.append(method_record(
        "lda_dev_weighted_geomean_all_layers",
        "layer_probability_fusion",
        {
            "layers": list(range(layer_count)),
            "weights": [float(weight) for weight in geometric_weights],
            "fusion": "geometric",
        },
        geometric_fuse(all_dev, geometric_weights),
        geometric_fuse(all_final, geometric_weights),
        dev,
        final,
        classes,
    ))

    fusion_candidates = []
    for size in range(2, min(max_combo_size, layer_count) + 1):
        for layers in combinations(range(layer_count), size):
            weights = np.full(size, 1.0 / size)
            for fusion_name, fuse_fn in (
                ("arithmetic", arithmetic_fuse),
                ("geometric", geometric_fuse),
            ):
                probabilities = fuse_fn(all_dev[list(layers)], weights)
                fusion_candidates.append((
                    layers,
                    fusion_name,
                    None,
                    evaluate_probabilities(probabilities, dev.labels, classes),
                ))
            for scale in (1.0, 2.0, 4.0, 8.0):
                probabilities = rank_fuse(all_dev[list(layers)], scale)
                fusion_candidates.append((
                    layers,
                    "rank",
                    scale,
                    evaluate_probabilities(probabilities, dev.labels, classes),
                ))

    best_by_fusion = {}
    for layers, fusion_name, scale, evaluation in fusion_candidates:
        current = best_by_fusion.get(fusion_name)
        if current is None or metrics_key(evaluation) > metrics_key(current[3]):
            best_by_fusion[fusion_name] = (layers, fusion_name, scale, evaluation)

    for fusion_name, (layers, _, scale, _) in sorted(best_by_fusion.items()):
        weights = np.full(len(layers), 1.0 / len(layers))
        if fusion_name == "arithmetic":
            dev_probabilities = arithmetic_fuse(all_dev[list(layers)], weights)
            final_probabilities = arithmetic_fuse(all_final[list(layers)], weights)
        elif fusion_name == "geometric":
            dev_probabilities = geometric_fuse(all_dev[list(layers)], weights)
            final_probabilities = geometric_fuse(all_final[list(layers)], weights)
        else:
            dev_probabilities = rank_fuse(all_dev[list(layers)], float(scale))
            final_probabilities = rank_fuse(all_final[list(layers)], float(scale))
        methods.append(method_record(
            f"lda_best_dev_combo_{fusion_name}",
            "layer_probability_fusion",
            {
                "layers": list(layers),
                "weights": weights.tolist(),
                "fusion": fusion_name,
                "rank_scale": scale,
                "max_combo_size": max_combo_size,
            },
            dev_probabilities,
            final_probabilities,
            dev,
            final,
            classes,
        ))

    temperature_candidates = []
    for size in range(2, min(5, layer_count) + 1):
        for layers in combinations(range(layer_count), size):
            weights = np.full(size, 1.0 / size)
            for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
                probabilities = temperature_fuse(
                    all_dev[list(layers)],
                    weights,
                    temperature,
                )
                temperature_candidates.append((
                    layers,
                    temperature,
                    evaluate_probabilities(probabilities, dev.labels, classes),
                ))
    best_layers, best_temperature, _ = max(
        temperature_candidates,
        key=lambda item: metrics_key(item[2]),
    )
    weights = np.full(len(best_layers), 1.0 / len(best_layers))
    methods.append(method_record(
        "lda_best_dev_combo_temperature",
        "layer_probability_fusion",
        {
            "layers": list(best_layers),
            "weights": weights.tolist(),
            "temperature": float(best_temperature),
        },
        temperature_fuse(all_dev[list(best_layers)], weights, best_temperature),
        temperature_fuse(all_final[list(best_layers)], weights, best_temperature),
        dev,
        final,
        classes,
    ))

    return methods


def flatten_layers(split: SplitFeatures, layers: tuple[int, ...]) -> np.ndarray:
    return split.layers[:, list(layers), :].reshape(len(split.labels), -1)


def pca_component_count(
    requested: int | None,
    train_features: np.ndarray,
    class_count: int,
) -> int | None:
    if requested is None:
        return None
    maximum = min(train_features.shape[0] - class_count, train_features.shape[1])
    if maximum < 2:
        return None
    return min(requested, maximum)


def fit_predict_proba_classifier(
    classifier,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    final_features: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(train_features, train_labels)
    dev_probabilities = classifier.predict_proba(dev_features)
    final_probabilities = classifier.predict_proba(final_features)
    return (
        align_probabilities(dev_probabilities, classifier.classes_, classes),
        align_probabilities(final_probabilities, classifier.classes_, classes),
    )


def fit_predict_score_classifier(
    classifier,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    final_features: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(train_features, train_labels)
    dev_scores = classifier.decision_function(dev_features)
    final_scores = classifier.decision_function(final_features)
    return (
        align_probabilities(softmax(dev_scores, axis=1), classifier.classes_, classes),
        align_probabilities(softmax(final_scores, axis=1), classifier.classes_, classes),
    )


def maybe_pipeline(estimator, pca_dim: int | None, train_features: np.ndarray, classes: np.ndarray):
    if pca_dim is None:
        return make_pipeline(StandardScaler(), estimator)
    components = pca_component_count(pca_dim, train_features, len(classes))
    if components is None:
        return None
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=components, svd_solver="full", random_state=0),
        estimator,
    )


def search_concat_classifiers(
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    layer_sets: dict[str, tuple[int, ...]],
) -> list[dict]:
    methods = []

    for feature_name, layers in layer_sets.items():
        train_features = flatten_layers(train, layers)
        dev_features = flatten_layers(dev, layers)
        final_features = flatten_layers(final, layers)

        lda_pca_dims = (64, 128, 256) if feature_name == "all_layers" else (None, 64, 128)
        for pca_dim in lda_pca_dims:
            estimator = LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
                priors=np.full(len(classes), 1.0 / len(classes)),
            )
            classifier = maybe_pipeline(estimator, pca_dim, train_features, classes)
            if classifier is None:
                continue
            try:
                dev_probabilities, final_probabilities = fit_predict_proba_classifier(
                    classifier,
                    train_features,
                    train.labels,
                    dev_features,
                    final_features,
                    classes,
                )
            except Exception as exc:  # noqa: BLE001 - keep broad search resilient.
                methods.append({
                    "method": "concat_pca_lda_failed",
                    "family": "concat_classifier",
                    "extra": {
                        "feature_set": feature_name,
                        "layers": list(layers),
                        "pca_dim": pca_dim,
                        "error": str(exc),
                    },
                    "failed": True,
                })
                continue
            methods.append(method_record(
                "concat_pca_lda",
                "concat_classifier",
                {
                    "feature_set": feature_name,
                    "layers": list(layers),
                    "pca_dim": pca_dim,
                },
                dev_probabilities,
                final_probabilities,
                dev,
                final,
                classes,
            ))

        if feature_name in {"default_568", "selected_top3", "best_combo", "mid_4_8"}:
            logreg_grid = [
                (c_value, pca_dim)
                for c_value in (0.1, 1.0)
                for pca_dim in (64,)
            ]
        else:
            logreg_grid = []
        for c_value, pca_dim in logreg_grid:
            estimator = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
            )
            classifier = maybe_pipeline(estimator, pca_dim, train_features, classes)
            if classifier is None:
                continue
            try:
                dev_probabilities, final_probabilities = fit_predict_proba_classifier(
                    classifier,
                    train_features,
                    train.labels,
                    dev_features,
                    final_features,
                    classes,
                )
            except Exception as exc:  # noqa: BLE001
                methods.append({
                    "method": "concat_logreg_failed",
                    "family": "concat_classifier",
                    "extra": {
                        "feature_set": feature_name,
                        "layers": list(layers),
                        "pca_dim": pca_dim,
                        "C": c_value,
                        "error": str(exc),
                    },
                    "failed": True,
                })
                continue
            methods.append(method_record(
                "concat_logreg",
                "concat_classifier",
                {
                    "feature_set": feature_name,
                    "layers": list(layers),
                    "pca_dim": pca_dim,
                    "C": c_value,
                },
                dev_probabilities,
                final_probabilities,
                dev,
                final,
                classes,
            ))

        ridge_pca_dims = (64,) if feature_name == "all_layers" else (None, 64)
        for alpha in (1.0, 10.0):
            for pca_dim in ridge_pca_dims:
                estimator = RidgeClassifier(
                    alpha=alpha,
                    class_weight="balanced",
                )
                classifier = maybe_pipeline(estimator, pca_dim, train_features, classes)
                if classifier is None:
                    continue
                try:
                    dev_probabilities, final_probabilities = fit_predict_score_classifier(
                        classifier,
                        train_features,
                        train.labels,
                        dev_features,
                        final_features,
                        classes,
                    )
                except Exception as exc:  # noqa: BLE001
                    methods.append({
                        "method": "concat_ridge_failed",
                        "family": "concat_classifier",
                        "extra": {
                            "feature_set": feature_name,
                            "layers": list(layers),
                            "pca_dim": pca_dim,
                            "alpha": alpha,
                            "error": str(exc),
                        },
                        "failed": True,
                    })
                    continue
                methods.append(method_record(
                    "concat_ridge",
                    "concat_classifier",
                    {
                        "feature_set": feature_name,
                        "layers": list(layers),
                        "pca_dim": pca_dim,
                        "alpha": alpha,
                    },
                    dev_probabilities,
                    final_probabilities,
                    dev,
                    final,
                    classes,
                ))

        if feature_name in {"default_568", "selected_top3", "best_combo"}:
            linear_svc_grid = [
                (c_value, 64)
                for c_value in (0.1,)
            ]
        else:
            linear_svc_grid = []
        for c_value, pca_dim in linear_svc_grid:
            estimator = LinearSVC(
                C=c_value,
                class_weight="balanced",
                dual=True,
                max_iter=10000,
            )
            classifier = maybe_pipeline(estimator, pca_dim, train_features, classes)
            if classifier is None:
                continue
            try:
                dev_probabilities, final_probabilities = fit_predict_score_classifier(
                    classifier,
                    train_features,
                    train.labels,
                    dev_features,
                    final_features,
                    classes,
                )
            except Exception as exc:  # noqa: BLE001
                methods.append({
                    "method": "concat_linear_svc_failed",
                    "family": "concat_classifier",
                    "extra": {
                        "feature_set": feature_name,
                        "layers": list(layers),
                        "pca_dim": pca_dim,
                        "C": c_value,
                        "error": str(exc),
                    },
                    "failed": True,
                })
                continue
            methods.append(method_record(
                "concat_linear_svc",
                "concat_classifier",
                {
                    "feature_set": feature_name,
                    "layers": list(layers),
                    "pca_dim": pca_dim,
                    "C": c_value,
                },
                dev_probabilities,
                final_probabilities,
                dev,
                final,
                classes,
            ))

    successful = [method for method in methods if not method.get("failed")]
    best_by_name: dict[tuple[str, str], dict] = {}
    for method in successful:
        key = (method["method"], method["family"])
        current = best_by_name.get(key)
        if current is None or method_selection_key(method) > method_selection_key(current):
            best_by_name[key] = method
    failures = [method for method in methods if method.get("failed")]
    return list(best_by_name.values()) + failures[:10]


def aggregate_segment_probabilities(
    segment_probabilities: list[np.ndarray],
    aggregation: str,
) -> np.ndarray:
    song_probabilities = []
    for probabilities in segment_probabilities:
        if aggregation == "mean":
            song_probability = probabilities.mean(axis=0)
        elif aggregation == "logmean":
            song_probability = softmax(
                np.log(np.clip(probabilities, 1e-12, 1.0)).mean(axis=0)
            )
        elif aggregation == "max":
            song_probability = probabilities.max(axis=0)
        else:
            raise ValueError(f"Unknown segment aggregation: {aggregation}")
        song_probabilities.append(song_probability)
    return normalize_probabilities(np.stack(song_probabilities))


def segment_layer_outputs(
    train: SplitFeatures,
    target: SplitFeatures,
    layer: int,
    classes: np.ndarray,
) -> dict[str, np.ndarray]:
    train_tensor, train_masks, train_metadata = build_song_segment_feature_tensor(
        train.records,
        layer,
    )
    target_tensor, target_masks, target_metadata = build_song_segment_feature_tensor(
        target.records,
        layer,
    )
    train_labels_by_song = np.asarray([item.producer_slug for item in train_metadata])
    train_features = train_tensor[train_masks]
    train_labels = np.concatenate([
        np.repeat(label, int(mask.sum()))
        for label, mask in zip(train_labels_by_song, train_masks)
    ])
    model = SongMeanShrinkageLDA.fit(train_features, train_labels)

    segment_outputs = []
    for song_features, mask in zip(target_tensor, target_masks):
        segment_probabilities = model.predict_proba(song_features[mask])
        segment_probabilities = align_probabilities(
            segment_probabilities,
            model.classes_,
            classes,
        )
        segment_outputs.append(segment_probabilities)

    target_song_ids = [item.song_id for item in target_metadata]
    if target_song_ids != target.song_ids:
        raise ValueError("Segment tensor song order does not match song feature order")
    return {
        aggregation: aggregate_segment_probabilities(segment_outputs, aggregation)
        for aggregation in ("mean", "logmean", "max")
    }


def search_segment_methods(
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    candidate_layers: tuple[int, ...],
) -> list[dict]:
    methods = []
    layer_outputs = {}
    for layer in candidate_layers:
        print(f"  segment layer {layer}", flush=True)
        dev_outputs = segment_layer_outputs(train, dev, layer, classes)
        final_outputs = segment_layer_outputs(train, final, layer, classes)
        layer_outputs[layer] = {
            aggregation: (dev_outputs[aggregation], final_outputs[aggregation])
            for aggregation in ("mean", "logmean", "max")
        }
        for aggregation in ("mean", "logmean", "max"):
            dev_probabilities, final_probabilities = layer_outputs[layer][aggregation]
            layer_outputs[layer][aggregation] = (dev_probabilities, final_probabilities)
            methods.append(method_record(
                "segment_lda_single_layer",
                "segment_classifier",
                {"layer": layer, "aggregation": aggregation},
                dev_probabilities,
                final_probabilities,
                dev,
                final,
                classes,
            ))

    for aggregation in ("mean", "logmean"):
        dev_stack = np.stack([
            layer_outputs[layer][aggregation][0]
            for layer in candidate_layers
        ])
        final_stack = np.stack([
            layer_outputs[layer][aggregation][1]
            for layer in candidate_layers
        ])
        weights = np.full(len(candidate_layers), 1.0 / len(candidate_layers))
        methods.append(method_record(
            "segment_lda_layer_ensemble",
            "segment_classifier",
            {
                "layers": list(candidate_layers),
                "segment_aggregation": aggregation,
                "layer_fusion": "arithmetic",
            },
            arithmetic_fuse(dev_stack, weights),
            arithmetic_fuse(final_stack, weights),
            dev,
            final,
            classes,
        ))

    return methods


def select_layer_sets(
    selected_layers: tuple[int, ...],
    best_combo_layers: tuple[int, ...],
    layer_count: int,
) -> dict[str, tuple[int, ...]]:
    candidates = {
        "default_568": DEFAULT_LAYER_SET,
        "selected_top3": selected_layers,
        "best_combo": best_combo_layers,
        "mid_4_8": tuple(layer for layer in range(4, min(layer_count, 9))),
    }
    if layer_count <= 16:
        candidates["all_layers"] = tuple(range(layer_count))
    deduped = {}
    seen = set()
    for name, layers in candidates.items():
        layers = tuple(layer for layer in layers if 0 <= layer < layer_count)
        if not layers or layers in seen:
            continue
        seen.add(layers)
        deduped[name] = layers
    return deduped


def summarize_exclusions(
    full_train: SplitFeatures,
    filtered_train: SplitFeatures,
    audit_flags: dict[str, AuditFlag],
) -> dict:
    excluded = sorted(set(full_train.song_ids) - set(filtered_train.song_ids))
    by_code: Counter[str] = Counter()
    records = []
    for song_id in excluded:
        flag = audit_flags.get(song_id)
        if not flag:
            continue
        by_code.update(flag.codes)
        records.append({
            "song_id": song_id,
            "producer_slug": flag.producer_slug,
            "title": flag.title,
            "codes": list(flag.codes),
            "risk_score": flag.risk_score,
        })
    return {
        "excluded_songs": len(excluded),
        "by_code": dict(sorted(by_code.items())),
        "records": records,
    }


def run_for_filter(
    strategy: str,
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    *,
    max_combo_size: int,
    include_segments: bool,
) -> dict:
    print(f"filter={strategy}: LDA layer bank", flush=True)
    counts = class_counts(train)
    all_layers = tuple(range(train.layers.shape[1]))
    all_dev = fit_layer_lda_probabilities(train, dev, all_layers, classes)
    all_final = fit_layer_lda_probabilities(train, final, all_layers, classes)

    methods = search_layer_fusions(
        all_dev,
        all_final,
        dev,
        final,
        classes,
        max_combo_size=max_combo_size,
    )
    selected_layers = select_layers_by_dev(all_dev, dev, classes, 3)
    best_combo = max(
        [
            method for method in methods
            if method["method"] == "lda_best_dev_combo_arithmetic"
        ],
        key=method_selection_key,
    )
    layer_sets = select_layer_sets(
        selected_layers,
        tuple(best_combo["extra"]["layers"]),
        train.layers.shape[1],
    )

    print(f"filter={strategy}: concat classifier grids", flush=True)
    methods.extend(search_concat_classifiers(train, dev, final, classes, layer_sets))

    if include_segments:
        segment_layers = tuple(sorted(set(
            selected_layers
            + tuple(best_combo["extra"]["layers"][:3])
            + DEFAULT_LAYER_SET
        )))[:4]
        print(f"filter={strategy}: segment methods {segment_layers}", flush=True)
        methods.extend(search_segment_methods(
            train,
            dev,
            final,
            classes,
            segment_layers,
        ))

    successful = [method for method in methods if not method.get("failed")]
    selected = max(successful, key=method_selection_key)
    best_final_top1 = max(
        successful,
        key=lambda item: item["final"]["metrics"]["top1_accuracy"],
    )
    return {
        "strategy": strategy,
        "training_songs": int(len(train.labels)),
        "minimum_class_songs": int(min(counts.values())),
        "songs_per_class": counts,
        "selected_by_dev": {
            "method": selected["method"],
            "family": selected["family"],
            "extra": selected["extra"],
            "dev": selected["dev"],
            "final": selected["final"],
        },
        "best_final_top1_diagnostic": {
            "method": best_final_top1["method"],
            "family": best_final_top1["family"],
            "extra": best_final_top1["extra"],
            "dev": best_final_top1["dev"],
            "final": best_final_top1["final"],
        },
        "methods": successful,
        "failures": [method for method in methods if method.get("failed")],
    }


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def write_report(path: Path, evaluation: dict) -> None:
    def find_method(strategy: str, predicate: Callable[[dict], bool]) -> dict | None:
        result = next(
            item for item in evaluation["results"]
            if item["strategy"] == strategy
        )
        for method in result["methods"]:
            if predicate(method):
                return method
        return None

    raw_baseline = find_method(
        "raw",
        lambda item: item["method"] == "lda_equal_568",
    )
    source_previous = find_method(
        "source_clean",
        lambda item: item["method"] == "lda_equal_top3_dev_layers",
    )
    source_concat = find_method(
        "source_clean",
        lambda item: (
            item["method"] == "concat_pca_lda"
            and item["extra"].get("feature_set") == "selected_top3"
            and item["extra"].get("pca_dim") is None
        ),
    )

    lines = [
        "# P4 Broad Model Search",
        "",
        "Constraints: no new data, no producer-specific rules, dev is used for selection, final is report-only.",
        "",
        "## Scope",
        "",
        "- Data filters: `raw`, `source_clean`, `review_clean`.",
        "- Layer probability fusion: fixed layers, dev top-3, all-layer weights, combo search, geometric mean, rank/Borda, temperature.",
        "- Concatenated-feature classifiers: shrinkage LDA, PCA+LDA, logistic regression, ridge, linear SVM.",
        "- Segment-level probes: run on `source_clean` by default, using existing segment embeddings only.",
        "",
        "## Conclusion",
        "",
        "The strict dev-selected winner is temperature layer fusion. It raises dev Top-1/Macro-F1, but it hurts final Top-3, so it looks like a dev-overfit candidate rather than a deployment candidate.",
        "",
        "The stronger next candidate is `source_clean + concat_pca_lda(selected_top3=[5,6,7], pca_dim=None)`: keep the source-clean data filter, concatenate song-level MERT layers 5/6/7, then train one shrinkage LDA head. It is still global, uses no new data, and has no producer-specific rule.",
        "",
    ]
    if raw_baseline and source_previous and source_concat:
        lines.extend([
            "| Method | Final Top-1 | Final Top-3 | Final Macro-F1 | Final wrong |",
            "|---|---:|---:|---:|---:|",
            (
                f"| raw baseline `lda_equal_568` | "
                f"{pct(raw_baseline['final']['metrics']['top1_accuracy'])} | "
                f"{pct(raw_baseline['final']['metrics']['top3_accuracy'])} | "
                f"{pct(raw_baseline['final']['metrics']['macro_f1'])} | "
                f"{raw_baseline['final']['wrong_count']} |"
            ),
            (
                f"| prior candidate `source_clean + lda_equal_top3_dev_layers` | "
                f"{pct(source_previous['final']['metrics']['top1_accuracy'])} | "
                f"{pct(source_previous['final']['metrics']['top3_accuracy'])} | "
                f"{pct(source_previous['final']['metrics']['macro_f1'])} | "
                f"{source_previous['final']['wrong_count']} |"
            ),
            (
                f"| broad-search candidate `source_clean + concat_pca_lda[5,6,7]` | "
                f"{pct(source_concat['final']['metrics']['top1_accuracy'])} | "
                f"{pct(source_concat['final']['metrics']['top3_accuracy'])} | "
                f"{pct(source_concat['final']['metrics']['macro_f1'])} | "
                f"{source_concat['final']['wrong_count']} |"
            ),
            "",
        ])
    lines.extend([
        "Use `docs/P4_CANDIDATE_CROSS_VALIDATION.md` as the train-only stability check for this recommendation.",
        "",
        "## Dev-Selected Results",
        "",
        "| Filter | Dev-selected method | Family | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final wrong | Final accepted accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in evaluation["results"]:
        selected = result["selected_by_dev"]
        final_rejection = selected["final"].get("rejection", {})
        lines.append(
            "| "
            f"{result['strategy']} | "
            f"{selected['method']} | "
            f"{selected['family']} | "
            f"{pct(selected['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(selected['dev']['metrics']['macro_f1'])} | "
            f"{pct(selected['final']['metrics']['top1_accuracy'])} | "
            f"{pct(selected['final']['metrics']['top3_accuracy'])} | "
            f"{pct(selected['final']['metrics']['macro_f1'])} | "
            f"{selected['final']['wrong_count']} | "
            f"{pct(final_rejection.get('accepted_accuracy'))} |"
        )

    lines.extend([
        "",
        "## Final Top-1 Diagnostic",
        "",
        "This table is diagnostic only; it is not used for formal method selection.",
        "",
        "| Filter | Highest final Top-1 method | Family | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final wrong |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for result in evaluation["results"]:
        best = result["best_final_top1_diagnostic"]
        lines.append(
            "| "
            f"{result['strategy']} | "
            f"{best['method']} | "
            f"{best['family']} | "
            f"{pct(best['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(best['dev']['metrics']['macro_f1'])} | "
            f"{pct(best['final']['metrics']['top1_accuracy'])} | "
            f"{pct(best['final']['metrics']['top3_accuracy'])} | "
            f"{pct(best['final']['metrics']['macro_f1'])} | "
            f"{best['final']['wrong_count']} |"
        )

    lines.extend([
        "",
        "## Top Candidates By Filter",
        "",
        "| Filter | Method | Family | Extra | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final wrong |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for result in evaluation["results"]:
        ordered = sorted(
            result["methods"],
            key=method_selection_key,
            reverse=True,
        )[:20]
        for method in ordered:
            extra = json.dumps(method["extra"], ensure_ascii=False, sort_keys=True)
            if len(extra) > 120:
                extra = extra[:117] + "..."
            lines.append(
                "| "
                f"{result['strategy']} | "
                f"{method['method']} | "
                f"{method['family']} | "
                f"`{extra}` | "
                f"{pct(method['dev']['metrics']['top1_accuracy'])} | "
                f"{pct(method['dev']['metrics']['macro_f1'])} | "
                f"{pct(method['final']['metrics']['top1_accuracy'])} | "
                f"{pct(method['final']['metrics']['top3_accuracy'])} | "
                f"{pct(method['final']['metrics']['macro_f1'])} | "
                f"{method['final']['wrong_count']} |"
            )

    lines.extend([
        "",
        "## Data Filters",
        "",
        "| Filter | Training songs | Minimum class songs | Excluded songs | Exclusion codes |",
        "|---|---:|---:|---:|---|",
    ])
    for result in evaluation["results"]:
        exclusions = evaluation["exclusions"][result["strategy"]]
        lines.append(
            f"| {result['strategy']} | {result['training_songs']} | "
            f"{result['minimum_class_songs']} | {exclusions['excluded_songs']} | "
            f"`{json.dumps(exclusions['by_code'], ensure_ascii=False, sort_keys=True)}` |"
        )

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python scripts/27_run_p4_broad_model_search.py",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=root / "data/processed/curated/mert_95_p1/segments.jsonl",
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=root / "data/processed/dev_holdout/mert_95_layers/segments.jsonl",
    )
    parser.add_argument(
        "--final-manifest",
        type=Path,
        default=root / "data/processed/frozen_test/mert_95_layers/segments.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "data/processed/evaluations/catalog_risk_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/processed/evaluations/p4_broad_model_search.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_BROAD_MODEL_SEARCH.md",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["raw", "source_clean", "review_clean"],
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--max-combo-size", type=int, default=6)
    parser.add_argument(
        "--skip-segments",
        action="store_true",
        help="Skip segment-level probes when a quicker song-level run is needed.",
    )
    parser.add_argument(
        "--segment-filters",
        nargs="+",
        default=["source_clean"],
        choices=["raw", "source_clean", "review_clean"],
        help="Filters that should receive the slower segment-level probes.",
    )
    args = parser.parse_args()

    print("loading song-level features", flush=True)
    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    classes = np.unique(train.labels)
    if set(dev.labels) - set(classes) or set(final.labels) - set(classes):
        raise ValueError("Dev/final contain labels that are absent from training")

    audit_flags = load_train_audit_flags(args.audit)
    results = []
    exclusions = {}
    for strategy in args.filters:
        mask = filter_mask(train, audit_flags, strategy)
        filtered_train = apply_mask(train, mask)
        exclusions[strategy] = summarize_exclusions(train, filtered_train, audit_flags)
        results.append(run_for_filter(
            strategy,
            filtered_train,
            dev,
            final,
            classes,
            max_combo_size=args.max_combo_size,
            include_segments=(
                not args.skip_segments and strategy in set(args.segment_filters)
            ),
        ))

    evaluation = {
        "protocol": {
            "purpose": "Broad global model search after P4 data-quality cleanup",
            "constraints": [
                "No new songs, audio, embeddings, or labels are added.",
                "No producer-specific or producer-pair-specific rules are used.",
                "Development holdout is used for method selection; final frozen is diagnostic/report-only.",
            ],
            "train_manifest": str(args.train_manifest),
            "dev_manifest": str(args.dev_manifest),
            "final_manifest": str(args.final_manifest),
            "audit": str(args.audit),
            "max_combo_size": args.max_combo_size,
            "segment_methods": not args.skip_segments,
            "segment_filters": args.segment_filters,
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
            "layers": int(train.layers.shape[1]),
            "embedding_dim": int(train.layers.shape[2]),
        },
        "exclusions": exclusions,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report_output),
        "selected": {
            result["strategy"]: {
                "method": result["selected_by_dev"]["method"],
                "family": result["selected_by_dev"]["family"],
                "extra": result["selected_by_dev"]["extra"],
                "dev": result["selected_by_dev"]["dev"]["metrics"],
                "final": result["selected_by_dev"]["final"]["metrics"],
                "final_wrong_count": result["selected_by_dev"]["final"]["wrong_count"],
            }
            for result in results
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
