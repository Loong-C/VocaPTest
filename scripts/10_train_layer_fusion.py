#!/usr/bin/env python
"""Nested grouped evaluation and training for nonnegative MERT layer fusion."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibration import TemperatureScaler, select_rejection_threshold
from vocaptest.models.layer_fusion import LayerFusionLDA, optimize_nonnegative_weights
from vocaptest.models.song_lda import SongMeanShrinkageLDA, build_catalog
from vocaptest.utils.paths import project_root


def load_display_names(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        item["slug"]: item.get("display_name", item["slug"])
        for item in config.get("producers", [])
    }


def fit_layer_models(
    features: np.ndarray,
    labels: np.ndarray,
) -> list:
    return [
        SongMeanShrinkageLDA.fit(features[:, layer], labels).estimator
        for layer in range(features.shape[1])
    ]


def layer_probabilities(
    estimators: list,
    features: np.ndarray,
) -> np.ndarray:
    return np.stack([
        estimator.predict_proba(features[:, layer])
        for layer, estimator in enumerate(estimators)
    ])


def inner_oof_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    splits: int,
    seed: int,
) -> np.ndarray:
    classes = np.unique(labels)
    output = np.zeros(
        (features.shape[1], len(features), len(classes)),
        dtype=np.float64,
    )
    splitter = StratifiedGroupKFold(
        n_splits=splits,
        shuffle=True,
        random_state=seed,
    )
    for train_indices, val_indices in splitter.split(features, labels, groups):
        estimators = fit_layer_models(features[train_indices], labels[train_indices])
        output[:, val_indices, :] = layer_probabilities(
            estimators,
            features[val_indices],
        )
    return output


def metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    predictions = classes[probabilities.argmax(axis=1)]
    ranks = np.argsort(probabilities, axis=1)[:, ::-1]
    true_indices = np.searchsorted(classes, labels)
    true_ranks = np.array([
        int(np.where(ranks[row] == true_index)[0][0]) + 1
        for row, true_index in enumerate(true_indices)
    ])
    return {
        "top1_accuracy": float(np.mean(predictions == labels)),
        "top3_accuracy": float(np.mean(true_ranks <= 3)),
        "macro_f1": float(f1_score(
            labels,
            predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )),
        "mrr": float(np.mean(1.0 / true_ranks)),
    }


def aggregate_repeats(repeats: list[dict], key: str) -> dict:
    values = np.array([item[key] for item in repeats])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=root / "data/processed/curated/mert_95_p1/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--model-output",
        default=root / "data/processed/models/p1_layer_fusion_lda.pkl",
        type=Path,
    )
    parser.add_argument(
        "--evaluation-output",
        default=root / "data/processed/evaluations/p1_layer_fusion.json",
        type=Path,
    )
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=0.8)
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    features, metadata = build_song_layer_feature_matrix(records, mode="mean")
    labels = np.array([item.producer_slug for item in metadata])
    groups = np.array([item.work_id for item in metadata])
    classes = np.unique(labels)
    true_indices = np.searchsorted(classes, labels)
    n_layers = features.shape[1]

    repeat_metrics: list[dict] = []
    per_layer_repeat_metrics = [[] for _ in range(n_layers)]
    nested_probabilities: list[np.ndarray] = []
    nested_true_indices: list[np.ndarray] = []
    outer_weights: list[np.ndarray] = []

    for repeat in range(args.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=args.outer_splits,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        fused_oof = np.zeros((len(features), len(classes)), dtype=np.float64)
        layer_oof = np.zeros(
            (n_layers, len(features), len(classes)),
            dtype=np.float64,
        )
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(features, labels, groups)
        ):
            inner_probabilities = inner_oof_probabilities(
                features[train_indices],
                labels[train_indices],
                groups[train_indices],
                splits=args.inner_splits,
                seed=args.seed + repeat * 100 + fold,
            )
            inner_true = np.searchsorted(classes, labels[train_indices])
            weights = optimize_nonnegative_weights(inner_probabilities, inner_true)
            outer_weights.append(weights)

            estimators = fit_layer_models(features[train_indices], labels[train_indices])
            test_layer_probabilities = layer_probabilities(
                estimators,
                features[test_indices],
            )
            layer_oof[:, test_indices, :] = test_layer_probabilities
            fused_oof[test_indices] = np.tensordot(
                weights,
                test_layer_probabilities,
                axes=(0, 0),
            )

        repeat_metrics.append(metrics(fused_oof, labels, classes))
        for layer in range(n_layers):
            per_layer_repeat_metrics[layer].append(
                metrics(layer_oof[layer], labels, classes)
            )
        nested_probabilities.append(fused_oof)
        nested_true_indices.append(true_indices)

    flat_probabilities = np.concatenate(nested_probabilities)
    flat_true_indices = np.concatenate(nested_true_indices)
    calibrator = TemperatureScaler.fit(flat_probabilities, flat_true_indices)
    calibrated = calibrator.transform(flat_probabilities)
    rejection = select_rejection_threshold(
        calibrated,
        flat_true_indices,
        target_precision=args.target_precision,
        minimum_coverage=0.1,
    )

    final_oof = inner_oof_probabilities(
        features,
        labels,
        groups,
        splits=args.outer_splits,
        seed=args.seed + 10000,
    )
    final_weights = optimize_nonnegative_weights(final_oof, true_indices)
    final_estimators = fit_layer_models(features, labels)
    display_names = load_display_names(root / "configs/producers.yaml")
    model = LayerFusionLDA(
        estimators=final_estimators,
        layer_weights=final_weights,
        temperature_scaler=calibrator,
        rejection_threshold=rejection["threshold"],
        display_names=display_names,
        catalog=build_catalog(
            labels.tolist(),
            [item.segment_count for item in metadata],
        ),
    )
    model.save(args.model_output)

    metric_names = ["top1_accuracy", "top3_accuracy", "macro_f1", "mrr"]
    evaluation = {
        "protocol": {
            "outer": f"{args.outer_repeats}x{args.outer_splits} StratifiedGroupKFold",
            "inner_splits": args.inner_splits,
            "group_key": "work_id",
            "weight_constraint": "nonnegative_sum_to_one",
            "pooling": "uniform_segments_then_song_mean",
        },
        "dataset": {
            "songs": len(metadata),
            "segments": int(sum(item.segment_count for item in metadata)),
            "producers": len(classes),
            "layers": n_layers,
            "embedding_dim": int(features.shape[2]),
            "songs_per_producer": dict(sorted(Counter(labels).items())),
        },
        "fusion": {
            "aggregate": {
                name: aggregate_repeats(repeat_metrics, name)
                for name in metric_names
            },
            "repeat_metrics": repeat_metrics,
            "final_layer_weights": final_weights.tolist(),
            "mean_outer_layer_weights": np.mean(outer_weights, axis=0).tolist(),
        },
        "layers": [
            {
                "layer": layer,
                "aggregate": {
                    name: aggregate_repeats(per_layer_repeat_metrics[layer], name)
                    for name in metric_names
                },
            }
            for layer in range(n_layers)
        ],
        "calibration": {
            "temperature": calibrator.temperature,
            "rejection": rejection,
            "target_precision": args.target_precision,
        },
    }
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.evaluation_output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)

    best_layer = max(
        evaluation["layers"],
        key=lambda item: item["aggregate"]["top1_accuracy"]["mean"],
    )
    print(json.dumps({
        "fusion": evaluation["fusion"]["aggregate"],
        "best_layer": best_layer,
        "final_layer_weights": final_weights.tolist(),
        "calibration": evaluation["calibration"],
        "model_output": str(args.model_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
