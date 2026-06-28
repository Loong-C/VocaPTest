#!/usr/bin/env python
"""Train the deployable P4 calibrated-stacking model artifact."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibrated_stacking import (
    CalibratedStackingLDA,
    _meta_features,
    _normalize,
)
from vocaptest.models.calibration import (
    confidence_signals,
    select_rejection_threshold,
)
from vocaptest.models.song_lda import SongMeanShrinkageLDA, build_catalog
from vocaptest.utils.paths import project_root


METRIC_NAMES = ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr", "log_loss")


@dataclass(frozen=True)
class SplitFeatures:
    layers: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    song_ids: list[str]
    segment_counts: list[int]


@dataclass(frozen=True)
class AuditFlag:
    song_id: str
    codes: tuple[str, ...]


def repo_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def load_display_names(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        item["slug"]: item.get("display_name", item["slug"])
        for item in config.get("producers", [])
    }


def load_split(path: Path) -> SplitFeatures:
    records = load_embedding_manifest(path)
    layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
    return SplitFeatures(
        layers=layers,
        labels=np.asarray([item.producer_slug for item in metadata]),
        groups=np.asarray([item.work_id for item in metadata]),
        song_ids=[item.song_id for item in metadata],
        segment_counts=[item.segment_count for item in metadata],
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
    flags = {}
    for record in audit["records"]:
        if record.get("split") != "train" or not record.get("flags"):
            continue
        song_id = source_key(record)
        if not song_id:
            continue
        flags[song_id] = AuditFlag(
            song_id=song_id,
            codes=tuple(flag["code"] for flag in record["flags"]),
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
    blocked = codes_by_strategy[strategy]
    keep = []
    for song_id in split.song_ids:
        flag = audit_flags.get(song_id)
        keep.append(flag is None or not bool(set(flag.codes) & blocked))
    return np.asarray(keep, dtype=bool)


def apply_mask(split: SplitFeatures, mask: np.ndarray) -> SplitFeatures:
    return SplitFeatures(
        layers=split.layers[mask],
        labels=split.labels[mask],
        groups=split.groups[mask],
        song_ids=[song_id for song_id, keep in zip(split.song_ids, mask) if keep],
        segment_counts=[
            count for count, keep in zip(split.segment_counts, mask) if keep
        ],
    )


def base_specs() -> list[dict]:
    return [
        {"name": "layer_6", "kind": "layer", "layers": [6]},
        {"name": "layer_7", "kind": "layer", "layers": [7]},
        {"name": "layer_8", "kind": "layer", "layers": [8]},
        {"name": "fusion_567", "kind": "layer_fusion", "layers": [5, 6, 7]},
        {"name": "fusion_568", "kind": "layer_fusion", "layers": [5, 6, 8]},
        {"name": "concat_56", "kind": "concat", "layers": [5, 6]},
        {"name": "concat_67", "kind": "concat", "layers": [6, 7]},
        {"name": "concat_567", "kind": "concat", "layers": [5, 6, 7]},
        {"name": "concat_568", "kind": "concat", "layers": [5, 6, 8]},
        {"name": "concat_678", "kind": "concat", "layers": [6, 7, 8]},
        {"name": "concat_789", "kind": "concat", "layers": [7, 8, 9]},
        {"name": "concat_45678", "kind": "concat", "layers": [4, 5, 6, 7, 8]},
    ]


def flatten(layer_features: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    return layer_features[:, list(layers), :].reshape(layer_features.shape[0], -1)


def align_probabilities(
    probabilities: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(target_classes)), dtype=np.float64)
    target_lookup = {str(label): index for index, label in enumerate(target_classes)}
    for source_index, label in enumerate(source_classes):
        aligned[:, target_lookup[str(label)]] = probabilities[:, source_index]
    return _normalize(aligned)


def fit_base_head(spec: dict, train_layers: np.ndarray, train_labels: np.ndarray) -> dict:
    layers = tuple(spec["layers"])
    if spec["kind"] == "layer":
        model = SongMeanShrinkageLDA.fit(train_layers[:, layers[0], :], train_labels)
        return {**spec, "estimator": model.estimator}
    if spec["kind"] == "layer_fusion":
        return {
            **spec,
            "estimators": [
                SongMeanShrinkageLDA.fit(train_layers[:, layer, :], train_labels).estimator
                for layer in layers
            ],
        }
    if spec["kind"] == "concat":
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(len(np.unique(train_labels)), 1.0 / len(np.unique(train_labels))),
        )
        estimator.fit(flatten(train_layers, layers), train_labels)
        return {**spec, "estimator": estimator}
    raise ValueError(f"Unknown base kind: {spec['kind']}")


def predict_base_head(
    head: dict,
    layer_features: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    kind = head["kind"]
    layers = tuple(head["layers"])
    if kind == "layer":
        estimator = head["estimator"]
        return align_probabilities(
            estimator.predict_proba(layer_features[:, layers[0], :]),
            estimator.classes_,
            classes,
        )
    if kind == "layer_fusion":
        outputs = [
            align_probabilities(
                estimator.predict_proba(layer_features[:, layer, :]),
                estimator.classes_,
                classes,
            )
            for layer, estimator in zip(layers, head["estimators"])
        ]
        return _normalize(np.mean(np.stack(outputs), axis=0))
    if kind == "concat":
        estimator = head["estimator"]
        return align_probabilities(
            estimator.predict_proba(flatten(layer_features, layers)),
            estimator.classes_,
            classes,
        )
    raise ValueError(f"Unknown base kind: {kind}")


def build_base_probabilities(
    train: SplitFeatures,
    target_layers: np.ndarray,
    classes: np.ndarray,
    specs: list[dict],
) -> np.ndarray:
    blocks = []
    for spec in specs:
        head = fit_base_head(spec, train.layers, train.labels)
        blocks.append(predict_base_head(head, target_layers, classes))
    return np.stack(blocks, axis=1)


def build_oof_probabilities(
    train: SplitFeatures,
    classes: np.ndarray,
    specs: list[dict],
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    blocks = []
    for index, spec in enumerate(specs, start=1):
        print(f"base {index}/{len(specs)} {spec['name']}", flush=True)
        probabilities = np.zeros((len(train.labels), len(classes)), dtype=np.float64)
        for fit_indices, validation_indices in splitter.split(
            train.layers,
            train.labels,
            train.groups,
        ):
            fold_train = SplitFeatures(
                layers=train.layers[fit_indices],
                labels=train.labels[fit_indices],
                groups=train.groups[fit_indices],
                song_ids=[],
                segment_counts=[],
            )
            head = fit_base_head(spec, fold_train.layers, fold_train.labels)
            probabilities[validation_indices] = predict_base_head(
                head,
                train.layers[validation_indices],
                classes,
            )
        blocks.append(probabilities)
    return np.stack(blocks, axis=1)


def fit_final_base_heads(
    train: SplitFeatures,
    specs: list[dict],
) -> list[dict]:
    heads = []
    for spec in specs:
        heads.append(fit_base_head(spec, train.layers, train.labels))
    return heads


def strip_inference_unused_state(heads: list[dict]) -> int:
    """Remove large sklearn training attributes that are not used by predict_proba."""
    removed = 0
    for head in heads:
        estimators = head.get("estimators") or [head.get("estimator")]
        for estimator in estimators:
            if estimator is not None and hasattr(estimator, "covariance_"):
                delattr(estimator, "covariance_")
                removed += 1
    return removed


def fit_meta_model(
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    *,
    feature_mode: str,
    c_value: float,
) -> tuple[LogisticRegression, StandardScaler]:
    features = _meta_features(train_probabilities, feature_mode)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(features, train_labels)
    return model, scaler


def predict_meta(
    model: LogisticRegression,
    scaler: StandardScaler,
    probabilities: np.ndarray,
    classes: np.ndarray,
    feature_mode: str,
) -> np.ndarray:
    features = scaler.transform(_meta_features(probabilities, feature_mode))
    return align_probabilities(model.predict_proba(features), model.classes_, classes)


def meta_cv_probabilities(
    train_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    *,
    feature_mode: str,
    c_value: float,
    folds: int,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros((len(labels), len(classes)), dtype=np.float64)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    for fit_indices, validation_indices in splitter.split(
        train_probabilities,
        labels,
        groups,
    ):
        model, scaler = fit_meta_model(
            train_probabilities[fit_indices],
            labels[fit_indices],
            feature_mode=feature_mode,
            c_value=c_value,
        )
        probabilities[validation_indices] = predict_meta(
            model,
            scaler,
            train_probabilities[validation_indices],
            classes,
            feature_mode,
        )
    return probabilities


def metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    probabilities = _normalize(probabilities)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = classes[order[:, 0]]
    lookup = {str(label): i for i, label in enumerate(classes)}
    ranks = np.asarray([
        int(np.where(order[row] == lookup[str(label)])[0][0]) + 1
        for row, label in enumerate(labels)
    ])
    true_indices = np.asarray([lookup[str(label)] for label in labels])
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
        "log_loss": float(log_loss(
            true_indices,
            probabilities,
            labels=np.arange(len(classes)),
        )),
    }


def rejection_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> dict:
    lookup = {str(label): i for i, label in enumerate(classes)}
    true_indices = np.asarray([lookup[str(label)] for label in labels])
    predictions = probabilities.argmax(axis=1)
    signals = confidence_signals(probabilities)
    accepted = signals["confidence"] >= threshold
    correct = predictions == true_indices
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
    }


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        default=root / "data/processed/curated/mert_95_p1/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--dev-manifest",
        default=root / "data/processed/dev_holdout/mert_95_layers/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--final-manifest",
        default=root / "data/processed/frozen_test/mert_95_layers/segments.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--audit",
        default=root / "data/processed/evaluations/catalog_risk_audit.json",
        type=Path,
    )
    parser.add_argument(
        "--model-output",
        default=root / "data/processed/models/p4_calibrated_stacking.pkl",
        type=Path,
    )
    parser.add_argument(
        "--evaluation-output",
        default=root / "data/processed/evaluations/p4_calibrated_stacking_deploy.json",
        type=Path,
    )
    parser.add_argument(
        "--strategy",
        default="source_clean",
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--base-folds", type=int, default=5)
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--meta-feature-mode", default="log_prob")
    parser.add_argument("--meta-c", type=float, default=0.03)
    parser.add_argument("--target-precision", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("loading features", flush=True)
    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    audit_flags = load_train_audit_flags(args.audit)
    train = apply_mask(train, filter_mask(train, audit_flags, args.strategy))
    classes = np.unique(train.labels)
    specs = base_specs()

    print("building OOF base probabilities", flush=True)
    train_probabilities = build_oof_probabilities(
        train,
        classes,
        specs,
        folds=args.base_folds,
        seed=args.seed,
    )
    print("fitting meta model", flush=True)
    meta_model, meta_scaler = fit_meta_model(
        train_probabilities,
        train.labels,
        feature_mode=args.meta_feature_mode,
        c_value=args.meta_c,
    )
    train_cv_probabilities = meta_cv_probabilities(
        train_probabilities,
        train.labels,
        train.groups,
        classes,
        feature_mode=args.meta_feature_mode,
        c_value=args.meta_c,
        folds=args.meta_folds,
        seed=args.seed + 4100,
    )
    train_true_indices = np.asarray([
        int(np.where(classes == label)[0][0])
        for label in train.labels
    ])
    rejection = select_rejection_threshold(
        train_cv_probabilities,
        train_true_indices,
        target_precision=args.target_precision,
        minimum_coverage=0.1,
    )

    print("fitting final base heads", flush=True)
    final_heads = fit_final_base_heads(train, specs)
    stripped_attributes = strip_inference_unused_state(final_heads)
    model = CalibratedStackingLDA(
        base_heads=final_heads,
        meta_model=meta_model,
        meta_scaler=meta_scaler,
        rejection_threshold=rejection["threshold"],
        meta_feature_mode=args.meta_feature_mode,
        display_names=load_display_names(root / "configs/producers.yaml"),
        catalog=build_catalog(train.labels.tolist(), train.segment_counts),
        embedding_backend="mert_95_p4_calibrated_stacking",
    )
    model.save(args.model_output)

    dev_probabilities = model.predict_proba_from_layer_features(dev.layers)
    final_probabilities = model.predict_proba_from_layer_features(final.layers)
    evaluation = {
        "protocol": {
            "purpose": "Deployable P4 calibrated-stacking artifact",
            "train_manifest": repo_path(args.train_manifest, root),
            "dev_manifest": repo_path(args.dev_manifest, root),
            "final_manifest": repo_path(args.final_manifest, root),
            "strategy": args.strategy,
            "base_heads": [
                {key: value for key, value in spec.items() if key != "estimator"}
                for spec in specs
            ],
            "base_oof": f"{args.base_folds} StratifiedGroupKFold by work_id",
            "meta_cv": f"{args.meta_folds} StratifiedGroupKFold by work_id",
            "meta_model": {
                "kind": "logistic_regression",
                "C": args.meta_c,
                "feature_mode": args.meta_feature_mode,
                "class_weight": "balanced",
            },
            "target_precision": args.target_precision,
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "removed_train_songs": int(len(load_split(args.train_manifest).labels) - len(train.labels)),
            "train_segments": int(sum(train.segment_counts)),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
            "minimum_class_songs": int(min(Counter(train.labels).values())),
        },
        "train_cv": {
            "metrics": metrics(train_cv_probabilities, train.labels, classes),
            "rejection": rejection,
            "rejection_metrics": rejection_metrics(
                train_cv_probabilities,
                train.labels,
                classes,
                rejection["threshold"],
            ),
        },
        "dev": {
            "metrics": metrics(dev_probabilities, dev.labels, classes),
            "rejection_metrics": rejection_metrics(
                dev_probabilities,
                dev.labels,
                classes,
                rejection["threshold"],
            ),
        },
        "final": {
            "metrics": metrics(final_probabilities, final.labels, classes),
            "rejection_metrics": rejection_metrics(
                final_probabilities,
                final.labels,
                classes,
                rejection["threshold"],
            ),
        },
        "artifact": {
            "model_output": repo_path(args.model_output, root),
            "backend": model.to_reference_library()["backend"],
            "classes": int(len(model.classes_)),
            "base_head_count": int(len(model.base_heads)),
            "meta_feature_mode": model.meta_feature_mode,
            "stripped_inference_unused_attributes": int(stripped_attributes),
        },
    }
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "model_output": str(args.model_output),
        "backend": model.to_reference_library()["backend"],
        "dev": evaluation["dev"]["metrics"],
        "final": evaluation["final"]["metrics"],
        "rejection": evaluation["train_cv"]["rejection_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
