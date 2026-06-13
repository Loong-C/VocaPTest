#!/usr/bin/env python
"""Nested grouped comparison of RDA, dual prototypes, and attention pooling."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import (
    build_song_layer_feature_matrix,
    build_song_segment_feature_tensor,
)
from vocaptest.models.attention_song_pooling import (
    fit_attention_song_classifier,
    pool_attention_song_features,
    predict_attention_song_classifier,
)
from vocaptest.models.discriminant_variants import (
    DualPrototypeClassifier,
    RegularizedDiscriminantClassifier,
)
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.utils.paths import project_root


METRIC_NAMES = ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr")


def metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> dict:
    predictions = classes[probabilities.argmax(axis=1)]
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    true_indices = np.searchsorted(classes, labels)
    ranks = np.array([
        int(np.where(order[row] == true_index)[0][0]) + 1
        for row, true_index in enumerate(true_indices)
    ])
    return {
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
    }


def aggregate(items: list[dict]) -> dict:
    output = {}
    for name in METRIC_NAMES:
        values = np.asarray([item[name] for item in items])
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def fit_lda_projection(
    features: np.ndarray,
    labels: np.ndarray,
) -> LinearDiscriminantAnalysis:
    classes = np.unique(labels)
    model = LinearDiscriminantAnalysis(
        solver="eigen",
        shrinkage="auto",
        priors=np.full(len(classes), 1.0 / len(classes)),
        n_components=len(classes) - 1,
    )
    return model.fit(features, labels)


def select_rda_parameters(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    inner_splits: int,
) -> dict:
    dimensions = (16, 32, 64)
    class_weights = (0.0, 0.5, 1.0)
    isotropic_weights = (0.1, 0.5, 0.9)
    candidates = [
        {
            "pca_dim": dimension,
            "covariance_structure": "full",
            "class_covariance_weight": class_weight,
            "isotropic_weight": isotropic_weight,
        }
        for dimension in dimensions
        for class_weight in class_weights
        for isotropic_weight in isotropic_weights
    ]
    candidates.extend([
        {
            "pca_dim": None,
            "covariance_structure": "diagonal",
            "class_covariance_weight": class_weight,
            "isotropic_weight": isotropic_weight,
        }
        for class_weight in class_weights
        for isotropic_weight in isotropic_weights
    ])
    classes = np.unique(labels)
    candidate_probabilities = [
        np.zeros((len(features), len(classes)), dtype=np.float64)
        for _ in candidates
    ]
    splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    for train_indices, validation_indices in splitter.split(
        features,
        labels,
        groups,
    ):
        pca = PCA(n_components=max(dimensions), svd_solver="full")
        train_projected = pca.fit_transform(features[train_indices])
        validation_projected = pca.transform(features[validation_indices])
        for index, candidate in enumerate(candidates):
            dimension = candidate["pca_dim"]
            train_candidate = (
                features[train_indices]
                if dimension is None
                else train_projected[:, :dimension]
            )
            validation_candidate = (
                features[validation_indices]
                if dimension is None
                else validation_projected[:, :dimension]
            )
            model = RegularizedDiscriminantClassifier.fit(
                train_candidate,
                labels[train_indices],
                class_covariance_weight=candidate["class_covariance_weight"],
                isotropic_weight=candidate["isotropic_weight"],
                diagonal=candidate["covariance_structure"] == "diagonal",
            )
            candidate_probabilities[index][validation_indices] = model.predict_proba(
                validation_candidate
            )

    return max(
        zip(candidates, candidate_probabilities),
        key=lambda item: (
            metrics(item[1], labels, classes)["macro_f1"],
            metrics(item[1], labels, classes)["top1_accuracy"],
        ),
    )[0]


def predict_rda(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    parameters: dict,
) -> np.ndarray:
    dimension = parameters["pca_dim"]
    if dimension is None:
        train_projected = train_features
        test_projected = test_features
    else:
        pca = PCA(n_components=dimension, svd_solver="full")
        train_projected = pca.fit_transform(train_features)
        test_projected = pca.transform(test_features)
    model = RegularizedDiscriminantClassifier.fit(
        train_projected,
        train_labels,
        class_covariance_weight=parameters["class_covariance_weight"],
        isotropic_weight=parameters["isotropic_weight"],
        diagonal=parameters["covariance_structure"] == "diagonal",
    )
    return model.predict_proba(test_projected)


def select_prototype_parameters(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    inner_splits: int,
) -> dict:
    dimensions = (8, 12, 19)
    standardization = (False, True)
    distance_scales = (0.5, 1.0, 2.0)
    candidates = [
        {
            "discriminant_dim": dimension,
            "standardize": standardize,
            "distance_scale": distance_scale,
        }
        for dimension in dimensions
        for standardize in standardization
        for distance_scale in distance_scales
    ]
    candidate_indices = {
        (
            candidate["discriminant_dim"],
            candidate["standardize"],
            candidate["distance_scale"],
        ): index
        for index, candidate in enumerate(candidates)
    }
    classes = np.unique(labels)
    candidate_probabilities = [
        np.zeros((len(features), len(classes)), dtype=np.float64)
        for _ in candidates
    ]
    splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels, groups)
    ):
        projection = fit_lda_projection(features[train_indices], labels[train_indices])
        train_projected = projection.transform(features[train_indices])
        validation_projected = projection.transform(features[validation_indices])
        for dimension in dimensions:
            for standardize in standardization:
                model = DualPrototypeClassifier.fit(
                    train_projected[:, :dimension],
                    labels[train_indices],
                    standardize=standardize,
                    distance_scale=1.0,
                    seed=seed + fold * 100,
                )
                for distance_scale in distance_scales:
                    model.distance_scale = distance_scale
                    index = candidate_indices[
                        (dimension, standardize, distance_scale)
                    ]
                    candidate_probabilities[index][validation_indices] = (
                        model.predict_proba(
                            validation_projected[:, :dimension]
                        )
                    )

    return max(
        zip(candidates, candidate_probabilities),
        key=lambda item: (
            metrics(item[1], labels, classes)["macro_f1"],
            metrics(item[1], labels, classes)["top1_accuracy"],
        ),
    )[0]


def predict_dual_prototype(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    parameters: dict,
    *,
    seed: int,
) -> np.ndarray:
    dimension = parameters["discriminant_dim"]
    projection = fit_lda_projection(train_features, train_labels)
    train_projected = projection.transform(train_features)[:, :dimension]
    test_projected = projection.transform(test_features)[:, :dimension]
    model = DualPrototypeClassifier.fit(
        train_projected,
        train_labels,
        standardize=parameters["standardize"],
        distance_scale=parameters["distance_scale"],
        seed=seed,
    )
    return model.predict_proba(test_projected)


def normalized_attention_entropy(
    attention: np.ndarray,
    masks: np.ndarray,
) -> tuple[float, float]:
    entropies = []
    peaks = []
    for weights, mask in zip(attention, masks):
        valid = weights[mask]
        peaks.append(float(valid.max()))
        if len(valid) <= 1:
            entropies.append(1.0)
        else:
            entropy = -float(np.sum(valid * np.log(valid + 1e-12)))
            entropies.append(entropy / np.log(len(valid)))
    return float(np.mean(entropies)), float(np.mean(peaks))


def config_key(parameters: dict) -> str:
    return json.dumps(parameters, sort_keys=True)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=root / "data/processed/curated/mert_95_p1/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=root / "data/processed/evaluations/p1_model_experiments.json",
        type=Path,
    )
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    all_layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
    mean_features = all_layers[:, args.layer, :]
    segment_features, segment_masks, segment_metadata = (
        build_song_segment_feature_tensor(records, args.layer)
    )
    if [item.song_id for item in metadata] != [
        item.song_id for item in segment_metadata
    ]:
        raise ValueError("Mean and segment feature order does not match")

    labels = np.asarray([item.producer_slug for item in metadata])
    groups = np.asarray([item.work_id for item in metadata])
    classes = np.unique(labels)
    integer_labels = np.searchsorted(classes, labels)
    method_repeat_metrics = {
        "baseline_shrinkage_lda": [],
        "regularized_discriminant": [],
        "dual_prototype": [],
        "attention_pooling_lda": [],
        "attention_pooling_linear": [],
    }
    rda_selections: Counter[str] = Counter()
    prototype_selections: Counter[str] = Counter()
    attention_epochs = []
    attention_entropies = []
    attention_peaks = []

    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=args.splits,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        probabilities = {
            method: np.zeros((len(labels), len(classes)), dtype=np.float64)
            for method in method_repeat_metrics
        }
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(mean_features, labels, groups)
        ):
            fold_seed = args.seed + repeat * 1000 + fold * 100
            baseline = SongMeanShrinkageLDA.fit(
                mean_features[train_indices],
                labels[train_indices],
            )
            probabilities["baseline_shrinkage_lda"][test_indices] = (
                baseline.predict_proba(mean_features[test_indices])
            )

            rda_parameters = select_rda_parameters(
                mean_features[train_indices],
                labels[train_indices],
                groups[train_indices],
                seed=fold_seed,
                inner_splits=args.inner_splits,
            )
            rda_selections[config_key(rda_parameters)] += 1
            probabilities["regularized_discriminant"][test_indices] = predict_rda(
                mean_features[train_indices],
                labels[train_indices],
                mean_features[test_indices],
                rda_parameters,
            )

            prototype_parameters = select_prototype_parameters(
                mean_features[train_indices],
                labels[train_indices],
                groups[train_indices],
                seed=fold_seed + 20,
                inner_splits=args.inner_splits,
            )
            prototype_selections[config_key(prototype_parameters)] += 1
            probabilities["dual_prototype"][test_indices] = (
                predict_dual_prototype(
                    mean_features[train_indices],
                    labels[train_indices],
                    mean_features[test_indices],
                    prototype_parameters,
                    seed=fold_seed + 40,
                )
            )

            attention_model, selected_epoch = fit_attention_song_classifier(
                segment_features[train_indices],
                segment_masks[train_indices],
                integer_labels[train_indices],
                groups[train_indices],
                class_count=len(classes),
                seed=fold_seed + 60,
                device=args.device,
            )
            attention_epochs.append(selected_epoch)
            train_pooled, _ = pool_attention_song_features(
                attention_model,
                segment_features[train_indices],
                segment_masks[train_indices],
                device=args.device,
            )
            test_pooled, test_attention = pool_attention_song_features(
                attention_model,
                segment_features[test_indices],
                segment_masks[test_indices],
                device=args.device,
            )
            attention_lda = SongMeanShrinkageLDA.fit(
                train_pooled,
                labels[train_indices],
            )
            probabilities["attention_pooling_lda"][test_indices] = (
                attention_lda.predict_proba(test_pooled)
            )
            linear_probabilities, _ = predict_attention_song_classifier(
                attention_model,
                segment_features[test_indices],
                segment_masks[test_indices],
                device=args.device,
            )
            probabilities["attention_pooling_linear"][test_indices] = (
                linear_probabilities
            )
            entropy, peak = normalized_attention_entropy(
                test_attention,
                segment_masks[test_indices],
            )
            attention_entropies.append(entropy)
            attention_peaks.append(peak)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"repeat={repeat + 1}/{args.repeats} "
                f"fold={fold + 1}/{args.splits} "
                f"rda={config_key(rda_parameters)} "
                f"prototype={config_key(prototype_parameters)} "
                f"attention_epoch={selected_epoch}",
                flush=True,
            )

        for method, method_probabilities in probabilities.items():
            method_repeat_metrics[method].append(
                metrics(method_probabilities, labels, classes)
            )

    evaluation = {
        "protocol": {
            "outer": f"{args.repeats}x{args.splits} StratifiedGroupKFold",
            "inner_splits": args.inner_splits,
            "group_key": "work_id",
            "selected_layer": args.layer,
            "selection_metric": "inner macro_f1, then top1_accuracy",
            "mert_frozen": True,
        },
        "dataset": {
            "songs": len(labels),
            "segments": int(segment_masks.sum()),
            "classes": len(classes),
            "embedding_dim": int(mean_features.shape[1]),
        },
        "methods": {
            method: {
                "aggregate": aggregate(repeat_metrics),
                "repeat_metrics": repeat_metrics,
            }
            for method, repeat_metrics in method_repeat_metrics.items()
        },
        "selection_diagnostics": {
            "rda": dict(rda_selections.most_common()),
            "dual_prototype": dict(prototype_selections.most_common()),
            "attention": {
                "selected_epoch_mean": float(np.mean(attention_epochs)),
                "selected_epoch_min": int(np.min(attention_epochs)),
                "selected_epoch_max": int(np.max(attention_epochs)),
                "normalized_entropy_mean": float(np.mean(attention_entropies)),
                "peak_weight_mean": float(np.mean(attention_peaks)),
                "parameter_count": int(sum(
                    parameter.numel()
                    for parameter in attention_model.parameters()
                )),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        method: result["aggregate"]
        for method, result in evaluation["methods"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
