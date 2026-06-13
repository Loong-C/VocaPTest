#!/usr/bin/env python
"""Evaluate the deployed P1 model on the untouched frozen song catalog."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibration import confidence_signals
from vocaptest.models.layer_fusion import LayerFusionLDA
from vocaptest.utils.paths import project_root


def ranking_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = classes[order[:, 0]]
    class_index = {str(slug): index for index, slug in enumerate(classes)}
    ranks = np.array([
        int(np.where(order[row] == class_index[str(label)])[0][0]) + 1
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
        },
        predictions,
        ranks,
    )


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            root
            / "data"
            / "processed"
            / "frozen_test"
            / "mert_95_layers"
            / "segments.jsonl"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            root
            / "data"
            / "processed"
            / "models"
            / "p1_selected_layer_lda.pkl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "data"
            / "processed"
            / "evaluations"
            / "p1_frozen_test.json"
        ),
    )
    args = parser.parse_args()

    records = load_embedding_manifest(args.manifest.resolve())
    all_layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
    model = LayerFusionLDA.load(args.model.resolve())
    labels = np.array([item.producer_slug for item in metadata])
    unknown = set(labels).difference(str(item) for item in model.classes_)
    if unknown:
        raise ValueError(f"Frozen labels missing from model: {sorted(unknown)}")
    counts = Counter(labels)
    if set(counts.values()) != {2}:
        raise ValueError(
            f"Expected exactly two frozen songs per class: {dict(counts)}"
        )

    probabilities = model.predict_proba_from_layer_features(all_layers)
    metrics, predictions, ranks = ranking_metrics(
        probabilities,
        labels,
        model.classes_,
    )
    signals = confidence_signals(probabilities)
    accepted = signals["confidence"] >= model.rejection_threshold
    correct = predictions == labels
    accepted_count = int(accepted.sum())

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=model.classes_,
        zero_division=0,
    )
    per_class = {
        str(slug): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "top3_accuracy": float(np.mean(ranks[labels == slug] <= 3)),
            "accepted": int(accepted[labels == slug].sum()),
        }
        for index, slug in enumerate(model.classes_)
    }
    songs = []
    for index, item in enumerate(metadata):
        order = np.argsort(probabilities[index])[::-1][:3]
        songs.append({
            "song_id": item.song_id,
            "work_id": item.work_id,
            "title": item.title,
            "producer_slug": item.producer_slug,
            "prediction": str(predictions[index]),
            "rank": int(ranks[index]),
            "accepted": bool(accepted[index]),
            "confidence": float(signals["confidence"][index]),
            "margin": float(signals["margin"][index]),
            "entropy": float(signals["entropy"][index]),
            "top3": [
                {
                    "producer_slug": str(model.classes_[class_index]),
                    "score": float(probabilities[index, class_index]),
                }
                for class_index in order
            ],
        })

    evaluation = {
        "protocol": {
            "name": "strict_frozen_song_holdout",
            "training_overlap": False,
            "used_for_model_selection": False,
            "used_for_calibration": False,
            "pooling": "uniform_segments_then_song_mean",
            "selected_layer": model.layer_indices,
        },
        "dataset": {
            "songs": len(metadata),
            "segments": int(sum(item.segment_count for item in metadata)),
            "classes": len(counts),
            "songs_per_class": dict(sorted(counts.items())),
        },
        "metrics": metrics,
        "rejection": {
            "threshold": model.rejection_threshold,
            "coverage": float(np.mean(accepted)),
            "accepted_count": accepted_count,
            "accepted_accuracy": (
                float(np.mean(correct[accepted]))
                if accepted_count
                else None
            ),
            "rejected_correct_count": int(np.sum(correct & ~accepted)),
        },
        "per_class": per_class,
        "songs": songs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
    print(json.dumps(
        {
            "dataset": evaluation["dataset"],
            "metrics": evaluation["metrics"],
            "rejection": evaluation["rejection"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
