#!/usr/bin/env python
"""Evaluate a frozen-MERT regularized projection and contrastive head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.projection_head import (
    fit_projection_head,
    predict_projection_head,
)
from vocaptest.utils.paths import project_root


def metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    true_indices = np.searchsorted(classes, labels)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    ranks = np.array([
        int(np.where(order[row] == true_index)[0][0]) + 1
        for row, true_index in enumerate(true_indices)
    ])
    return {
        "top1_accuracy": float(np.mean(predictions == true_indices)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "macro_f1": float(f1_score(
            true_indices,
            predictions,
            labels=np.arange(len(classes)),
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
        "--output",
        default=root / "data/processed/evaluations/p1_projection_head.json",
        type=Path,
    )
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    all_layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
    features = all_layers[:, args.layer, :]
    text_labels = np.array([item.producer_slug for item in metadata])
    groups = np.array([item.work_id for item in metadata])
    classes = np.unique(text_labels)
    labels = np.searchsorted(classes, text_labels)
    repeat_results = []
    selected_epochs = []

    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=args.splits,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        probabilities = np.zeros((len(features), len(classes)))
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(features, text_labels, groups)
        ):
            model, scaler, best_epoch = fit_projection_head(
                features[train_indices],
                labels[train_indices],
                groups[train_indices],
                class_count=len(classes),
                seed=args.seed + repeat * 100 + fold,
                device=args.device,
                projection_dim=args.projection_dim,
            )
            selected_epochs.append(best_epoch)
            probabilities[test_indices] = predict_projection_head(
                model,
                scaler,
                features[test_indices],
                args.device,
            )
        repeat_results.append(metrics(probabilities, text_labels, classes))

    aggregate = {}
    for name in ["top1_accuracy", "top3_accuracy", "macro_f1", "mrr"]:
        values = np.array([item[name] for item in repeat_results])
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
            "projection_dim": args.projection_dim,
            "balanced_sampler": True,
            "supervised_contrastive_weight": 0.1,
        },
        "aggregate": aggregate,
        "repeat_metrics": repeat_results,
        "selected_epochs": {
            "mean": float(np.mean(selected_epochs)),
            "min": int(np.min(selected_epochs)),
            "max": int(np.max(selected_epochs)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
