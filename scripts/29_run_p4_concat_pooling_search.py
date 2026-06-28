#!/usr/bin/env python
"""Non-cheating search over pooled layer concatenation classifiers."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocaptest.data.curation import load_embedding_manifest  # noqa: E402
from vocaptest.features.layer_features import build_song_layer_feature_matrix  # noqa: E402
from vocaptest.utils.paths import project_root  # noqa: E402


RAW_BASELINE_TOP1 = 0.7842105263157895
RAW_BASELINE_TOP3 = 0.8894736842105263
RAW_BASELINE_MACRO_F1 = 0.7554502164502164
TARGET_DELTA = 0.04
TARGET_TOP1 = RAW_BASELINE_TOP1 + TARGET_DELTA
TARGET_TOP3 = RAW_BASELINE_TOP3 + 0.03
TARGET_MACRO_F1 = RAW_BASELINE_MACRO_F1 + 0.04


def success_gates(final_metrics: dict) -> dict[str, bool]:
    top1 = final_metrics["top1_accuracy"]
    top3 = final_metrics["top3_accuracy"]
    macro = final_metrics["macro_f1"]
    return {
        "top1_plus_4pp_guarded": (
            top1 >= TARGET_TOP1
            and top3 >= RAW_BASELINE_TOP3 - 0.005
            and macro >= RAW_BASELINE_MACRO_F1 + 0.02
        ),
        "top3_plus_3pp_guarded": (
            top3 >= TARGET_TOP3
            and top1 >= RAW_BASELINE_TOP1 + 0.02
            and macro >= RAW_BASELINE_MACRO_F1 + 0.02
        ),
        "macro_f1_plus_4pp_guarded": (
            macro >= TARGET_MACRO_F1
            and top1 >= RAW_BASELINE_TOP1 + 0.02
            and top3 >= RAW_BASELINE_TOP3 - 0.005
        ),
        "balanced_plus_3_1p5_3pp": (
            top1 >= RAW_BASELINE_TOP1 + 0.03
            and top3 >= RAW_BASELINE_TOP3 + 0.015
            and macro >= RAW_BASELINE_MACRO_F1 + 0.03
        ),
    }


def target_met(final_metrics: dict) -> bool:
    return any(success_gates(final_metrics).values())


@dataclass(frozen=True)
class SplitFeatures:
    layers: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    song_ids: list[str]


@dataclass(frozen=True)
class AuditFlag:
    song_id: str
    codes: tuple[str, ...]


def load_split(path: Path, *, mode: str) -> SplitFeatures:
    records = load_embedding_manifest(path)
    layers, metadata = build_song_layer_feature_matrix(records, mode=mode)
    return SplitFeatures(
        layers=layers,
        labels=np.asarray([item.producer_slug for item in metadata]),
        groups=np.asarray([item.work_id for item in metadata]),
        song_ids=[item.song_id for item in metadata],
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
    )


def normalize(probabilities: np.ndarray) -> np.ndarray:
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
    return normalize(aligned)


def metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    probabilities = normalize(probabilities)
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


def selection_key(item: dict) -> tuple[float, float, float, float]:
    dev = item["dev"]["metrics"]
    return (
        dev["macro_f1"],
        dev["top1_accuracy"],
        dev["top3_accuracy"],
        -dev["log_loss"],
    )


def flatten(split: SplitFeatures, layers: tuple[int, ...]) -> np.ndarray:
    return split.layers[:, list(layers), :].reshape(len(split.labels), -1)


def effective_pca_dim(
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


def make_estimator(
    kind: str,
    *,
    classes: np.ndarray,
    pca_dim: int | None,
    standardize: bool,
    train_features: np.ndarray,
    params: dict,
):
    if kind == "lda":
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(len(classes), 1.0 / len(classes)),
        )
    elif kind == "logreg":
        estimator = LogisticRegression(
            C=float(params["C"]),
            class_weight="balanced",
            max_iter=1200,
            solver="lbfgs",
        )
    elif kind == "ridge":
        estimator = RidgeClassifier(
            alpha=float(params["alpha"]),
            class_weight="balanced",
        )
    elif kind == "linear_svc":
        estimator = LinearSVC(
            C=float(params["C"]),
            class_weight="balanced",
            dual=True,
            max_iter=12000,
        )
    else:
        raise ValueError(f"Unknown estimator kind: {kind}")

    steps = []
    if standardize:
        steps.append(StandardScaler())
    pca_components = effective_pca_dim(pca_dim, train_features, len(classes))
    if pca_dim is not None and pca_components is None:
        return None
    if pca_components is not None:
        steps.append(PCA(
            n_components=pca_components,
            svd_solver="full",
            random_state=0,
        ))
    steps.append(estimator)
    return make_pipeline(*steps)


def fit_classifier(
    classifier,
    train_features: np.ndarray,
    train_labels: np.ndarray,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(train_features, train_labels)
    return classifier


def predict_classifier(
    classifier,
    kind: str,
    target_features: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    if kind in {"lda", "logreg"}:
        probabilities = classifier.predict_proba(target_features)
    else:
        scores = classifier.decision_function(target_features)
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
    estimator_classes = classifier.classes_
    return align_probabilities(probabilities, estimator_classes, classes)


def evaluate_candidate(
    candidate: dict,
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
) -> dict:
    layers = tuple(candidate["layers"])
    train_features = flatten(train, layers)
    dev_features = flatten(dev, layers)
    final_features = flatten(final, layers)
    classifier = make_estimator(
        candidate["kind"],
        classes=classes,
        pca_dim=candidate["pca_dim"],
        standardize=candidate["standardize"],
        train_features=train_features,
        params=candidate["params"],
    )
    if classifier is None:
        raise ValueError("Invalid PCA dimension")
    classifier = fit_classifier(
        classifier,
        train_features,
        train.labels,
    )
    dev_probabilities = predict_classifier(
        classifier,
        candidate["kind"],
        dev_features,
        classes,
    )
    final_probabilities = predict_classifier(
        classifier,
        candidate["kind"],
        final_features,
        classes,
    )
    return {
        "candidate": candidate,
        "dev": {
            "metrics": metrics(dev_probabilities, dev.labels, classes),
        },
        "final": {
            "metrics": metrics(final_probabilities, final.labels, classes),
        },
    }


def layer_windows(layer_count: int) -> list[tuple[int, ...]]:
    windows = []
    for width in (1, 2, 3, 5):
        for start in range(0, layer_count - width + 1):
            windows.append(tuple(range(start, start + width)))
    windows.extend([
        (5, 6, 7),
        (5, 6, 8),
        (4, 5, 6, 7, 8),
        (1, 5, 6, 7),
        (2, 3, 4, 6, 7, 11),
    ])
    deduped = []
    seen = set()
    for item in windows:
        item = tuple(layer for layer in item if 0 <= layer < layer_count)
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def candidate_grid(mode: str, layer_count: int) -> list[dict]:
    candidates = []
    known_full_lda = {
        (5, 6, 7),
        (5, 6, 8),
        (4, 5, 6, 7, 8),
    }
    for layers in layer_windows(layer_count):
        width = len(layers)
        layers_tuple = tuple(layers)
        if mode != "mean" and width > 5:
            continue
        allow_full_lda = (
            (mode == "mean" and (width <= 3 or layers_tuple in known_full_lda))
        )
        if allow_full_lda:
            candidates.append({
                "name": "concat_lda",
                "mode": mode,
                "kind": "lda",
                "layers": list(layers),
                "pca_dim": None,
                "standardize": False,
                "params": {},
            })
        if width >= 3:
            for pca_dim in (32, 64, 128):
                candidates.append({
                    "name": "concat_pca_lda",
                    "mode": mode,
                    "kind": "lda",
                    "layers": list(layers),
                    "pca_dim": pca_dim,
                    "standardize": True,
                    "params": {},
                })
        if mode == "mean" and layers_tuple in {
            (5, 6, 7),
            (5, 6, 8),
            (4, 5, 6, 7, 8),
            (1, 5, 6, 7),
        }:
            for kind, params in (
                ("logreg", {"C": 0.1}),
                ("ridge", {"alpha": 1.0}),
                ("ridge", {"alpha": 10.0}),
                ("linear_svc", {"C": 0.1}),
            ):
                candidates.append({
                    "name": f"concat_{kind}",
                    "mode": mode,
                    "kind": kind,
                    "layers": list(layers),
                    "pca_dim": 64 if kind != "ridge" else None,
                    "standardize": True,
                    "params": params,
                })
    return candidates


def aggregate_cv_metrics(items: list[dict]) -> dict:
    output = {}
    for name in ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr", "log_loss"):
        values = np.asarray([item[name] for item in items], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return output


def cross_validate_candidate(
    candidate: dict,
    train: SplitFeatures,
    classes: np.ndarray,
    *,
    repeats: int,
    folds: int,
    seed: int,
) -> dict:
    layers = tuple(candidate["layers"])
    features = flatten(train, layers)
    repeat_metrics = []
    for repeat in range(repeats):
        probabilities = np.zeros((len(train.labels), len(classes)), dtype=np.float64)
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fit_indices, validation_indices in splitter.split(
            features,
            train.labels,
            train.groups,
        ):
            classifier = make_estimator(
                candidate["kind"],
                classes=classes,
                pca_dim=candidate["pca_dim"],
                standardize=candidate["standardize"],
                train_features=features[fit_indices],
                params=candidate["params"],
            )
            if classifier is None:
                continue
            classifier = fit_classifier(
                classifier,
                features[fit_indices],
                train.labels[fit_indices],
            )
            probabilities[validation_indices] = predict_classifier(
                classifier,
                candidate["kind"],
                features[validation_indices],
                classes,
            )
        repeat_metrics.append(metrics(probabilities, train.labels, classes))
    return {
        "aggregate": aggregate_cv_metrics(repeat_metrics),
        "repeat_metrics": repeat_metrics,
    }


def cv_selection_key(item: dict) -> tuple[float, float, float, float]:
    dev = item["dev"]["metrics"]
    cv = item["cv"]["aggregate"]
    return (
        cv["macro_f1"]["mean"],
        cv["top1_accuracy"]["mean"],
        dev["macro_f1"],
        dev["top1_accuracy"],
    )


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(path: Path, evaluation: dict) -> None:
    selected = evaluation["selected_candidate"]
    gates = evaluation["success_gates"]
    lines = [
        "# P4 Concat Pooling Search",
        "",
        "Protocol: no new data, no producer-specific rules, no final-based tuning. Candidates are searched on dev and stabilized with train-only grouped CV; final is used only after selection.",
        "",
        "Success target: any one guarded gate may pass, but no single metric is allowed to improve while the others collapse.",
        "",
        f"- Top-1 gate: Top-1 >= {pct(TARGET_TOP1)}, Top-3 >= {pct(RAW_BASELINE_TOP3 - 0.005)}, Macro-F1 >= {pct(RAW_BASELINE_MACRO_F1 + 0.02)}.",
        f"- Top-3 gate: Top-3 >= {pct(TARGET_TOP3)}, Top-1 >= {pct(RAW_BASELINE_TOP1 + 0.02)}, Macro-F1 >= {pct(RAW_BASELINE_MACRO_F1 + 0.02)}.",
        f"- Macro gate: Macro-F1 >= {pct(TARGET_MACRO_F1)}, Top-1 >= {pct(RAW_BASELINE_TOP1 + 0.02)}, Top-3 >= {pct(RAW_BASELINE_TOP3 - 0.005)}.",
        f"- Balanced gate: Top-1 >= {pct(RAW_BASELINE_TOP1 + 0.03)}, Top-3 >= {pct(RAW_BASELINE_TOP3 + 0.015)}, Macro-F1 >= {pct(RAW_BASELINE_MACRO_F1 + 0.03)}.",
        "",
        "## Selected Candidate",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| name | `{selected['candidate']['name']}` |",
        f"| mode | `{selected['candidate']['mode']}` |",
        f"| kind | `{selected['candidate']['kind']}` |",
        f"| layers | `{selected['candidate']['layers']}` |",
        f"| pca_dim | `{selected['candidate']['pca_dim']}` |",
        f"| standardize | `{selected['candidate']['standardize']}` |",
        f"| CV Top-1 | {pct(selected['cv']['aggregate']['top1_accuracy']['mean'])} |",
        f"| CV Macro-F1 | {pct(selected['cv']['aggregate']['macro_f1']['mean'])} |",
        f"| Dev Top-1 | {pct(selected['dev']['metrics']['top1_accuracy'])} |",
        f"| Dev Macro-F1 | {pct(selected['dev']['metrics']['macro_f1'])} |",
        f"| Final Top-1 | {pct(selected['final']['metrics']['top1_accuracy'])} |",
        f"| Final Top-3 | {pct(selected['final']['metrics']['top3_accuracy'])} |",
        f"| Final Macro-F1 | {pct(selected['final']['metrics']['macro_f1'])} |",
        f"| Target met | `{evaluation['target_met']}` |",
        "",
        "## Success Gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    for name, passed in gates.items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.extend([
        "",
        "## Top CV-Checked Candidates",
        "",
        "| Rank | Name | Mode | Kind | Layers | PCA | Std | CV Top-1 | CV Macro-F1 | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 |",
        "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(evaluation["cv_checked"], start=1):
        candidate = item["candidate"]
        lines.append(
            f"| {index} | `{candidate['name']}` | `{candidate['mode']}` | "
            f"`{candidate['kind']}` | `{candidate['layers']}` | "
            f"{candidate['pca_dim']} | `{candidate['standardize']}` | "
            f"{pct(item['cv']['aggregate']['top1_accuracy']['mean'])} | "
            f"{pct(item['cv']['aggregate']['macro_f1']['mean'])} | "
            f"{pct(item['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(item['dev']['metrics']['macro_f1'])} | "
            f"{pct(item['final']['metrics']['top1_accuracy'])} | "
            f"{pct(item['final']['metrics']['top3_accuracy'])} |"
        )
    lines.extend([
        "",
        "## Best Dev Candidates Before CV",
        "",
        "| Rank | Name | Mode | Kind | Layers | PCA | Std | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 |",
        "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(evaluation["top_dev_candidates"], start=1):
        candidate = item["candidate"]
        lines.append(
            f"| {index} | `{candidate['name']}` | `{candidate['mode']}` | "
            f"`{candidate['kind']}` | `{candidate['layers']}` | "
            f"{candidate['pca_dim']} | `{candidate['standardize']}` | "
            f"{pct(item['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(item['dev']['metrics']['macro_f1'])} | "
            f"{pct(item['final']['metrics']['top1_accuracy'])} | "
            f"{pct(item['final']['metrics']['top3_accuracy'])} |"
        )
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python scripts/29_run_p4_concat_pooling_search.py",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
        default=root / "data/processed/evaluations/p4_concat_pooling_search.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_CONCAT_POOLING_SEARCH.md",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["mean", "mean_std", "mean_std_change", "multi_stat"],
        choices=["mean", "mean_std", "mean_std_change", "multi_stat"],
    )
    parser.add_argument(
        "--strategy",
        default="source_clean",
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--top-dev", type=int, default=24)
    parser.add_argument("--top-cv", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audit_flags = load_train_audit_flags(args.audit)
    all_results = []
    top_dev_pool = []
    selected_by_mode = {}
    classes = None
    filtered_train_by_mode = {}
    dev_song_count = None
    final_song_count = None

    for mode in args.modes:
        print(f"loading mode={mode}", flush=True)
        train = load_split(args.train_manifest, mode=mode)
        dev = load_split(args.dev_manifest, mode=mode)
        final = load_split(args.final_manifest, mode=mode)
        if classes is None:
            classes = np.unique(train.labels)
            dev_song_count = int(len(dev.labels))
            final_song_count = int(len(final.labels))
        mask = filter_mask(train, audit_flags, args.strategy)
        train = apply_mask(train, mask)
        filtered_train_by_mode[mode] = train
        candidates = candidate_grid(mode, train.layers.shape[1])
        print(f"  mode={mode} candidates={len(candidates)}", flush=True)
        mode_results = []
        for index, candidate in enumerate(candidates, start=1):
            if index % 50 == 0:
                print(f"  mode={mode} candidate={index}/{len(candidates)}", flush=True)
            try:
                result = evaluate_candidate(candidate, train, dev, final, classes)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "candidate": candidate,
                    "failed": True,
                    "error": str(exc),
                }
            mode_results.append(result)
        successful = [item for item in mode_results if not item.get("failed")]
        selected_by_mode[mode] = max(successful, key=selection_key)
        top_dev_pool.extend(sorted(successful, key=selection_key, reverse=True)[:args.top_dev])
        all_results.extend(mode_results)

    top_dev_candidates = sorted(top_dev_pool, key=selection_key, reverse=True)[:args.top_dev]
    cv_checked = []
    for index, result in enumerate(top_dev_candidates[:args.top_cv], start=1):
        candidate = result["candidate"]
        print(
            f"cv {index}/{args.top_cv}: {candidate['mode']} "
            f"{candidate['name']} {candidate['layers']}",
            flush=True,
        )
        cv = cross_validate_candidate(
            candidate,
            filtered_train_by_mode[candidate["mode"]],
            classes,
            repeats=args.cv_repeats,
            folds=args.cv_folds,
            seed=args.seed,
        )
        cv_checked.append({**result, "cv": cv})

    selected_candidate = max(cv_checked, key=cv_selection_key)
    gates = success_gates(selected_candidate["final"]["metrics"])
    evaluation = {
        "protocol": {
            "purpose": "Non-cheating concat/pooling search for larger P4 gains",
            "strategy": args.strategy,
            "targets": {
                "raw_baseline_top1": RAW_BASELINE_TOP1,
                "raw_baseline_top3": RAW_BASELINE_TOP3,
                "raw_baseline_macro_f1": RAW_BASELINE_MACRO_F1,
                "top1_gate": TARGET_TOP1,
                "top3_gate": TARGET_TOP3,
                "macro_f1_gate": TARGET_MACRO_F1,
                "target_delta": TARGET_DELTA,
            },
            "selection": (
                "Candidate grid is searched on dev. Top dev candidates receive "
                "train-only grouped CV. Final is reported after selection only."
            ),
            "cv": {
                "repeats": args.cv_repeats,
                "folds": args.cv_folds,
                "seed": args.seed,
            },
        },
        "dataset": {
            "strategy": args.strategy,
            "train_songs": int(len(next(iter(filtered_train_by_mode.values())).labels)),
            "minimum_class_songs": int(min(Counter(
                next(iter(filtered_train_by_mode.values())).labels
            ).values())),
            "dev_songs": dev_song_count,
            "final_songs": final_song_count,
            "classes": int(len(classes)),
        },
        "selected_by_mode": selected_by_mode,
        "top_dev_candidates": top_dev_candidates,
        "cv_checked": cv_checked,
        "selected_candidate": selected_candidate,
        "success_gates": gates,
        "target_met": bool(any(gates.values())),
        "all_results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        "selected": selected_candidate["candidate"],
        "cv": selected_candidate["cv"]["aggregate"],
        "dev": selected_candidate["dev"]["metrics"],
        "final": selected_candidate["final"]["metrics"],
        "success_gates": gates,
        "target_met": evaluation["target_met"],
        "output": str(args.output),
        "report": str(args.report_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
