#!/usr/bin/env python
"""Evaluate and train song-mean Shrinkage LDA on the curated dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.song_features import build_song_feature_matrix
from vocaptest.models.song_lda import SongMeanShrinkageLDA, build_catalog
from vocaptest.utils.paths import project_root


def load_display_names(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        producer["slug"]: producer.get("display_name", producer["slug"])
        for producer in config.get("producers", [])
    }


def evaluate_grouped_cv(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    repeats: int,
    seed: int,
) -> dict:
    groups_per_class = {
        label: len(set(groups[labels == label]))
        for label in np.unique(labels)
    }
    n_splits = min(requested_splits, min(groups_per_class.values()))
    if n_splits < 2:
        raise ValueError("At least two independent works per class are required")

    classes = np.unique(labels)
    all_true: list[str] = []
    all_pred: list[str] = []
    top3_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    repeat_metrics: list[dict] = []

    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        repeat_true: list[str] = []
        repeat_pred: list[str] = []
        repeat_top3: list[float] = []
        repeat_rr: list[float] = []

        for train_indices, test_indices in splitter.split(features, labels, groups):
            model = SongMeanShrinkageLDA.fit(
                features[train_indices],
                labels[train_indices],
            )
            probabilities = model.predict_proba(features[test_indices])
            order = np.argsort(probabilities, axis=1)[:, ::-1]
            predictions = model.classes_[order[:, 0]]

            for row, true_label in enumerate(labels[test_indices]):
                ranked_labels = model.classes_[order[row]]
                rank = int(np.where(ranked_labels == true_label)[0][0]) + 1
                repeat_true.append(str(true_label))
                repeat_pred.append(str(predictions[row]))
                repeat_top3.append(float(rank <= 3))
                repeat_rr.append(1.0 / rank)

        all_true.extend(repeat_true)
        all_pred.extend(repeat_pred)
        top3_hits.extend(repeat_top3)
        reciprocal_ranks.extend(repeat_rr)
        repeat_metrics.append({
            "repeat": repeat,
            "top1_accuracy": float(np.mean(np.array(repeat_true) == np.array(repeat_pred))),
            "top3_accuracy": float(np.mean(repeat_top3)),
            "macro_f1": float(f1_score(
                repeat_true, repeat_pred, labels=classes, average="macro", zero_division=0
            )),
            "mrr": float(np.mean(repeat_rr)),
        })

    metric_names = ["top1_accuracy", "top3_accuracy", "macro_f1", "mrr"]
    aggregate = {}
    for name in metric_names:
        values = np.array([item[name] for item in repeat_metrics])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    matrix = confusion_matrix(all_true, all_pred, labels=classes)
    return {
        "protocol": {
            "splitter": "StratifiedGroupKFold",
            "group_key": "work_id",
            "n_splits": n_splits,
            "repeats": repeats,
            "seed": seed,
            "equal_class_priors": True,
            "shrinkage": "auto",
        },
        "aggregate": aggregate,
        "repeat_metrics": repeat_metrics,
        "classes": classes.tolist(),
        "confusion_matrix": matrix.tolist(),
        "evaluated_predictions": len(all_true),
    }


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=root / "data/processed/curated/mert_95_p0_20s/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--producer-config",
        default=root / "configs/producers.yaml",
        type=Path,
    )
    parser.add_argument(
        "--model-output",
        default=root / "data/processed/models/song_mean_shrinkage_lda.pkl",
        type=Path,
    )
    parser.add_argument(
        "--evaluation-output",
        default=root / "data/processed/evaluations/song_mean_shrinkage_lda.json",
        type=Path,
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    features, metadata = build_song_feature_matrix(records)
    labels = np.array([item.producer_slug for item in metadata])
    groups = np.array([item.work_id for item in metadata])

    evaluation = evaluate_grouped_cv(
        features,
        labels,
        groups,
        requested_splits=args.splits,
        repeats=args.repeats,
        seed=args.seed,
    )
    evaluation["dataset"] = {
        "songs": len(metadata),
        "segments": int(sum(item.segment_count for item in metadata)),
        "producers": len(set(labels)),
        "embedding_dim": int(features.shape[1]),
        "songs_per_producer": dict(sorted(Counter(labels).items())),
    }
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.evaluation_output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)

    display_names = load_display_names(args.producer_config)
    catalog = build_catalog(
        labels.tolist(),
        [item.segment_count for item in metadata],
    )
    backend = records[0].model_backend if records else "unknown"
    final_model = SongMeanShrinkageLDA.fit(
        features,
        labels,
        display_names=display_names,
        catalog=catalog,
        embedding_backend=backend,
    )
    final_model.save(args.model_output)

    print(json.dumps({
        "dataset": evaluation["dataset"],
        "aggregate": evaluation["aggregate"],
        "model_output": str(args.model_output),
        "evaluation_output": str(args.evaluation_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
