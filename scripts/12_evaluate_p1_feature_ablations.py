#!/usr/bin/env python
"""Grouped CV ablation for song statistics and traditional MIR features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.utils.paths import project_root


def load_mir(path: Path) -> dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        return {
            row["song_id"]: np.asarray(row["values"], dtype=np.float32)
            for row in (json.loads(line) for line in handle if line.strip())
        }


def build_projected_estimator(
    train_samples: int,
    feature_dim: int,
    class_count: int,
) -> Pipeline:
    components = min(64, feature_dim, train_samples - class_count - 1)
    priors = np.full(class_count, 1.0 / class_count)
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=42,
        )),
        ("lda", LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=priors,
        )),
    ])


def fold_metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
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
    for name in ["top1_accuracy", "top3_accuracy", "macro_f1", "mrr"]:
        values = np.array([item[name] for item in items])
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=root / "data/processed/curated/mert_95_p1/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--mir",
        default=root / "data/processed/features/p1_mir_features.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=root / "data/processed/evaluations/p1_feature_ablations.json",
        type=Path,
    )
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest)
    modes = ["mean", "mean_std", "mean_std_change", "multi_stat"]
    matrices = {}
    metadata = None
    for mode in modes:
        matrix, current_metadata = build_song_layer_feature_matrix(records, mode=mode)
        matrices[mode] = matrix[:, args.layer, :]
        if metadata is None:
            metadata = current_metadata
        elif [item.song_id for item in metadata] != [
            item.song_id for item in current_metadata
        ]:
            raise ValueError("Song ordering changed across pooling modes")
    assert metadata is not None

    labels = np.array([item.producer_slug for item in metadata])
    groups = np.array([item.work_id for item in metadata])
    classes = np.unique(labels)
    mir_by_song = load_mir(args.mir)
    mir = np.stack([mir_by_song[item.song_id] for item in metadata])

    candidates = {
        "layer_mean_raw_lda": matrices["mean"],
        "layer_mean_pca64": matrices["mean"],
        "layer_mean_std_pca64": matrices["mean_std"],
        "layer_mean_std_change_pca64": matrices["mean_std_change"],
        "layer_multi_stat_pca64": matrices["multi_stat"],
        "layer_mean_plus_mir_pca64": np.concatenate([matrices["mean"], mir], axis=1),
        "mir_only_lda": mir,
    }
    raw_candidates = {"layer_mean_raw_lda", "mir_only_lda"}
    results = {}

    for name, features in candidates.items():
        repeat_results = []
        for repeat in range(args.repeats):
            splitter = StratifiedGroupKFold(
                n_splits=args.splits,
                shuffle=True,
                random_state=args.seed + repeat,
            )
            probabilities = np.zeros((len(features), len(classes)))
            for train_indices, test_indices in splitter.split(features, labels, groups):
                if name == "layer_mean_raw_lda":
                    estimator = SongMeanShrinkageLDA.fit(
                        features[train_indices],
                        labels[train_indices],
                    ).estimator
                elif name == "mir_only_lda":
                    estimator = Pipeline([
                        ("scale", StandardScaler()),
                        ("lda", LinearDiscriminantAnalysis(
                            solver="lsqr",
                            shrinkage="auto",
                            priors=np.full(len(classes), 1.0 / len(classes)),
                        )),
                    ])
                    estimator.fit(features[train_indices], labels[train_indices])
                else:
                    estimator = build_projected_estimator(
                        len(train_indices),
                        features.shape[1],
                        len(classes),
                    )
                    estimator.fit(features[train_indices], labels[train_indices])
                probabilities[test_indices] = estimator.predict_proba(features[test_indices])
            repeat_results.append(fold_metrics(probabilities, labels, classes))
        results[name] = {
            "feature_dim": int(features.shape[1]),
            "aggregate": aggregate(repeat_results),
            "repeat_metrics": repeat_results,
        }

    evaluation = {
        "protocol": {
            "splitter": f"{args.repeats}x{args.splits} StratifiedGroupKFold",
            "group_key": "work_id",
            "selected_mert_layer": args.layer,
            "projected_dim": 64,
        },
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
    ranking = sorted(
        (
            (name, item["aggregate"]["top1_accuracy"]["mean"])
            for name, item in results.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    print(json.dumps({"ranking": ranking, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
