#!/usr/bin/env python
"""Train the deployable P3 5/6/8 layer-fusion model.

This turns the P3 experiment recommendation into a production artifact that the
API can load through the existing LayerFusionLDA path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibration import (
    TemperatureScaler,
    confidence_signals,
    select_rejection_threshold,
)
from vocaptest.models.layer_fusion import LayerFusionLDA
from vocaptest.models.song_lda import SongMeanShrinkageLDA, build_catalog
from vocaptest.utils.paths import project_root


METRIC_NAMES = ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr")


def load_display_names(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        item["slug"]: item.get("display_name", item["slug"])
        for item in config.get("producers", [])
    }


def repo_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def ranking_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> dict:
    predictions = classes[probabilities.argmax(axis=1)]
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    true_indices = np.searchsorted(classes, labels)
    ranks = np.asarray([
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


def aggregate_metrics(items: Sequence[dict]) -> dict:
    aggregate = {}
    for name in METRIC_NAMES:
        values = np.asarray([item[name] for item in items], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def fit_predict_layer_ensemble(
    train_layers: np.ndarray,
    train_labels: np.ndarray,
    test_layers: np.ndarray,
    layer_indices: Sequence[int],
) -> np.ndarray:
    probabilities = None
    for layer in layer_indices:
        model = SongMeanShrinkageLDA.fit(train_layers[:, layer, :], train_labels)
        layer_probabilities = model.predict_proba(test_layers[:, layer, :])
        probabilities = (
            layer_probabilities
            if probabilities is None
            else probabilities + layer_probabilities
        )
    return probabilities / len(layer_indices)


def oof_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    layer_indices: Sequence[int],
    *,
    repeats: int,
    splits: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    blocks = []
    repeat_metrics = []
    for repeat in range(repeats):
        probabilities = np.zeros((len(labels), len(classes)), dtype=np.float64)
        splitter = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        for train_indices, validation_indices in splitter.split(
            features,
            labels,
            groups,
        ):
            probabilities[validation_indices] = fit_predict_layer_ensemble(
                features[train_indices],
                labels[train_indices],
                features[validation_indices],
                layer_indices,
            )
        repeat_metrics.append(ranking_metrics(probabilities, labels, classes))
        blocks.append(probabilities)
    return np.concatenate(blocks), repeat_metrics


def rejection_metrics(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> dict:
    predictions = classes[probabilities.argmax(axis=1)]
    signals = confidence_signals(probabilities)
    accepted = signals["confidence"] >= threshold
    correct = predictions == labels
    return {
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
        "target_precision_oof": float(np.mean(correct[accepted]))
        if np.any(accepted)
        else 0.0,
        "true_class_count": int(len(np.unique(true_indices))),
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
        default=root / "data/processed/evaluations/p3_layer_fusion_deploy.json",
        type=Path,
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 6, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=0.96)
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    features, metadata = build_song_layer_feature_matrix(records, mode="mean")
    labels = np.asarray([item.producer_slug for item in metadata])
    groups = np.asarray([item.work_id for item in metadata])
    classes = np.unique(labels)
    true_indices = np.searchsorted(classes, labels)
    layer_indices = tuple(args.layers)

    if any(layer < 0 or layer >= features.shape[1] for layer in layer_indices):
        raise ValueError(f"Layer indices must be within 0..{features.shape[1] - 1}")

    flat_oof, repeat_metrics = oof_probabilities(
        features,
        labels,
        groups,
        classes,
        layer_indices,
        repeats=args.repeats,
        splits=args.splits,
        seed=args.seed,
    )
    flat_true_indices = np.tile(true_indices, args.repeats)
    flat_labels = np.tile(labels, args.repeats)
    calibrator = TemperatureScaler.fit(flat_oof, flat_true_indices)
    calibrated_oof = calibrator.transform(flat_oof)
    rejection = select_rejection_threshold(
        calibrated_oof,
        flat_true_indices,
        target_precision=args.target_precision,
        minimum_coverage=0.1,
    )

    final_estimators = [
        SongMeanShrinkageLDA.fit(features[:, layer, :], labels).estimator
        for layer in layer_indices
    ]
    layer_weights = np.full(len(layer_indices), 1.0 / len(layer_indices))
    model = LayerFusionLDA(
        estimators=final_estimators,
        layer_weights=layer_weights,
        temperature_scaler=calibrator,
        rejection_threshold=rejection["threshold"],
        layer_indices=list(layer_indices),
        calibration_input="probabilities",
        display_names=load_display_names(root / "configs/producers.yaml"),
        catalog=build_catalog(
            labels.tolist(),
            [item.segment_count for item in metadata],
        ),
        embedding_backend="mert_95_p3_layers_5_6_8",
    )
    model.save(args.model_output)

    evaluation = {
        "protocol": {
            "purpose": "Deployable P3 layer-fusion artifact",
            "manifest": repo_path(args.manifest, root),
            "layers": list(layer_indices),
            "layer_weights": layer_weights.tolist(),
            "oof": f"{args.repeats}x{args.splits} StratifiedGroupKFold by work_id",
            "target_precision": args.target_precision,
            "calibration_input": "probabilities",
        },
        "dataset": {
            "songs": int(len(labels)),
            "segments": int(sum(item.segment_count for item in metadata)),
            "classes": int(len(classes)),
            "layers_available": int(features.shape[1]),
            "embedding_dim": int(features.shape[2]),
            "class_names": classes.tolist(),
        },
        "oof": {
            "aggregate": aggregate_metrics(repeat_metrics),
            "repeat_metrics": repeat_metrics,
            "temperature": calibrator.temperature,
            "rejection": rejection,
            "rejection_metrics": rejection_metrics(
                calibrated_oof,
                flat_true_indices,
                flat_labels,
                classes,
                rejection["threshold"],
            ),
        },
        "artifact": {
            "model_output": repo_path(args.model_output, root),
            "backend": model.to_reference_library()["backend"],
            "classes": int(len(model.classes_)),
            "layer_indices": model.layer_indices,
        },
    }
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "model_output": str(args.model_output),
        "layers": list(layer_indices),
        "classes": int(len(classes)),
        "backend": model.to_reference_library()["backend"],
        "oof_aggregate": evaluation["oof"]["aggregate"],
        "rejection": evaluation["oof"]["rejection_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
