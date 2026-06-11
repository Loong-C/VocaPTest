#!/usr/bin/env python
"""Train the selected P1 MERT layer with OOF calibration and rejection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibration import TemperatureScaler, select_rejection_threshold
from vocaptest.models.layer_fusion import LayerFusionLDA
from vocaptest.models.song_lda import SongMeanShrinkageLDA, build_catalog
from vocaptest.utils.paths import project_root


def load_display_names(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        item["slug"]: item.get("display_name", item["slug"])
        for item in config.get("producers", [])
    }


def classification_metrics(
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
        default=root / "data/processed/models/p1_selected_layer_lda.pkl",
        type=Path,
    )
    parser.add_argument(
        "--evaluation-output",
        default=root / "data/processed/evaluations/p1_selected_layer.json",
        type=Path,
    )
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=0.95)
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    all_layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
    features = all_layers[:, args.layer, :]
    labels = np.array([item.producer_slug for item in metadata])
    groups = np.array([item.work_id for item in metadata])
    classes = np.unique(labels)
    true_indices = np.searchsorted(classes, labels)

    repeat_metrics = []
    oof_probabilities = []
    oof_logits = []
    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=args.splits,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        probabilities = np.zeros((len(features), len(classes)))
        logits = np.zeros((len(features), len(classes)))
        for train_indices, test_indices in splitter.split(features, labels, groups):
            model = SongMeanShrinkageLDA.fit(
                features[train_indices],
                labels[train_indices],
            )
            probabilities[test_indices] = model.predict_proba(features[test_indices])
            logits[test_indices] = model.estimator.decision_function(features[test_indices])
        repeat_metrics.append(classification_metrics(probabilities, labels, classes))
        oof_probabilities.append(probabilities)
        oof_logits.append(logits)

    flat_probabilities = np.concatenate(oof_probabilities)
    flat_logits = np.concatenate(oof_logits)
    flat_true_indices = np.tile(true_indices, args.repeats)
    calibrator = TemperatureScaler.fit_logits(flat_logits, flat_true_indices)
    calibrated = calibrator.transform_logits(flat_logits)
    rejection = select_rejection_threshold(
        calibrated,
        flat_true_indices,
        target_precision=args.target_precision,
        minimum_coverage=0.1,
    )

    final_estimator = SongMeanShrinkageLDA.fit(features, labels).estimator
    deployed = LayerFusionLDA(
        estimators=[final_estimator],
        layer_weights=np.array([1.0]),
        temperature_scaler=calibrator,
        rejection_threshold=rejection["threshold"],
        layer_indices=[args.layer],
        calibration_input="logits",
        display_names=load_display_names(root / "configs/producers.yaml"),
        catalog=build_catalog(
            labels.tolist(),
            [item.segment_count for item in metadata],
        ),
        embedding_backend="mert_95_selected_layer",
    )
    deployed.save(args.model_output)

    aggregate = {}
    for name in ["top1_accuracy", "top3_accuracy", "macro_f1", "mrr"]:
        values = np.array([item[name] for item in repeat_metrics])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    evaluation = {
        "protocol": {
            "splitter": f"{args.repeats}x{args.splits} StratifiedGroupKFold",
            "group_key": "work_id",
            "selected_layer": args.layer,
            "pooling": "uniform_segments_then_song_mean",
        },
        "dataset": {
            "songs": len(metadata),
            "segments": int(sum(item.segment_count for item in metadata)),
            "classes": len(classes),
        },
        "aggregate": aggregate,
        "repeat_metrics": repeat_metrics,
        "calibration": {
            "temperature": calibrator.temperature,
            "uncalibrated_log_loss": float(log_loss(
                flat_true_indices,
                flat_probabilities,
                labels=np.arange(len(classes)),
            )),
            "calibrated_log_loss": float(log_loss(
                flat_true_indices,
                calibrated,
                labels=np.arange(len(classes)),
            )),
            "target_precision": args.target_precision,
            "rejection": rejection,
        },
    }
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.evaluation_output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
