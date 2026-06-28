#!/usr/bin/env python
"""Global similarity classifiers under the P4 no-cheating protocol."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


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
TARGET_TOP1 = RAW_BASELINE_TOP1 + 0.04
TARGET_TOP3 = RAW_BASELINE_TOP3 + 0.03
TARGET_MACRO_F1 = RAW_BASELINE_MACRO_F1 + 0.04


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


def load_split(path: Path) -> SplitFeatures:
    records = load_embedding_manifest(path)
    layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
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


def flatten(split: SplitFeatures, layers: tuple[int, ...]) -> np.ndarray:
    return split.layers[:, list(layers), :].reshape(len(split.labels), -1)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def fit_transform_features(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    target_features: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    if transform == "raw_l2":
        return l2_normalize(train_features), l2_normalize(target_features)
    if transform == "standard_l2":
        scaler = StandardScaler().fit(train_features)
        return (
            l2_normalize(scaler.transform(train_features)),
            l2_normalize(scaler.transform(target_features)),
        )
    if transform.startswith("pca"):
        dim = int(transform[3:])
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        target_scaled = scaler.transform(target_features)
        pca = PCA(
            n_components=min(dim, train_scaled.shape[0] - len(np.unique(train_labels)), train_scaled.shape[1]),
            svd_solver="full",
            random_state=0,
        ).fit(train_scaled)
        return l2_normalize(pca.transform(train_scaled)), l2_normalize(pca.transform(target_scaled))
    if transform.startswith("lda"):
        dim = int(transform[3:])
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        target_scaled = scaler.transform(target_features)
        model = LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage="auto",
            priors=np.full(len(np.unique(train_labels)), 1.0 / len(np.unique(train_labels))),
            n_components=min(dim, len(np.unique(train_labels)) - 1),
        ).fit(train_scaled, train_labels)
        return l2_normalize(model.transform(train_scaled)), l2_normalize(model.transform(target_scaled))
    raise ValueError(f"Unknown transform: {transform}")


def transform_feature_triplet(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    final_features: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if transform == "raw_l2":
        return (
            l2_normalize(train_features),
            l2_normalize(dev_features),
            l2_normalize(final_features),
        )
    if transform == "standard_l2":
        scaler = StandardScaler().fit(train_features)
        return (
            l2_normalize(scaler.transform(train_features)),
            l2_normalize(scaler.transform(dev_features)),
            l2_normalize(scaler.transform(final_features)),
        )
    if transform.startswith("pca"):
        dim = int(transform[3:])
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        dev_scaled = scaler.transform(dev_features)
        final_scaled = scaler.transform(final_features)
        pca = PCA(
            n_components=min(
                dim,
                train_scaled.shape[0] - len(np.unique(train_labels)),
                train_scaled.shape[1],
            ),
            svd_solver="full",
            random_state=0,
        ).fit(train_scaled)
        return (
            l2_normalize(pca.transform(train_scaled)),
            l2_normalize(pca.transform(dev_scaled)),
            l2_normalize(pca.transform(final_scaled)),
        )
    if transform.startswith("lda"):
        dim = int(transform[3:])
        labels = np.unique(train_labels)
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        dev_scaled = scaler.transform(dev_features)
        final_scaled = scaler.transform(final_features)
        model = LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage="auto",
            priors=np.full(len(labels), 1.0 / len(labels)),
            n_components=min(dim, len(labels) - 1),
        ).fit(train_scaled, train_labels)
        return (
            l2_normalize(model.transform(train_scaled)),
            l2_normalize(model.transform(dev_scaled)),
            l2_normalize(model.transform(final_scaled)),
        )
    raise ValueError(f"Unknown transform: {transform}")


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def class_probability_from_scores(
    sample_scores: np.ndarray,
    train_labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    output = np.zeros((sample_scores.shape[0], len(classes)), dtype=np.float64)
    for index, label in enumerate(classes):
        output[:, index] = sample_scores[:, train_labels == label].sum(axis=1)
    return normalize_probabilities(output)


def predict_similarity(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    target_features: np.ndarray,
    classes: np.ndarray,
    candidate: dict,
) -> np.ndarray:
    similarities = target_features @ train_features.T
    if candidate["method"] == "knn":
        k = min(int(candidate["k"]), train_features.shape[0])
        top_indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        if candidate["weighted"]:
            weights = softmax(top_scores / float(candidate["temperature"]), axis=1)
        else:
            weights = np.full_like(top_scores, 1.0 / k)
        sample_scores = np.zeros_like(similarities)
        np.put_along_axis(sample_scores, top_indices, weights, axis=1)
        return class_probability_from_scores(sample_scores, train_labels, classes)
    if candidate["method"] == "soft_vote":
        weights = softmax(similarities / float(candidate["temperature"]), axis=1)
        return class_probability_from_scores(weights, train_labels, classes)
    if candidate["method"] == "centroid":
        centroids = []
        for label in classes:
            centroid = train_features[train_labels == label].mean(axis=0)
            centroids.append(centroid)
        centroid_features = l2_normalize(np.stack(centroids))
        return softmax(float(candidate["scale"]) * (target_features @ centroid_features.T), axis=1)
    raise ValueError(f"Unknown method: {candidate['method']}")


def metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    probabilities = normalize_probabilities(probabilities)
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


def evaluate_candidate(
    candidate: dict,
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
) -> dict:
    layers = tuple(candidate["layers"])
    train_base = flatten(train, layers)
    dev_base = flatten(dev, layers)
    final_base = flatten(final, layers)
    train_features, dev_features = fit_transform_features(
        train_base,
        train.labels,
        dev_base,
        candidate["transform"],
    )
    _, final_features = fit_transform_features(
        train_base,
        train.labels,
        final_base,
        candidate["transform"],
    )
    dev_probabilities = predict_similarity(
        train_features,
        train.labels,
        dev_features,
        classes,
        candidate,
    )
    final_probabilities = predict_similarity(
        train_features,
        train.labels,
        final_features,
        classes,
        candidate,
    )
    return {
        "candidate": candidate,
        "dev": {"metrics": metrics(dev_probabilities, dev.labels, classes)},
        "final": {"metrics": metrics(final_probabilities, final.labels, classes)},
    }


def selection_key(item: dict) -> tuple[float, float, float, float]:
    dev = item["dev"]["metrics"]
    return (
        dev["macro_f1"],
        dev["top1_accuracy"],
        dev["top3_accuracy"],
        -dev["log_loss"],
    )


def cv_selection_key(item: dict) -> tuple[float, float, float, float]:
    cv = item["cv"]["aggregate"]
    dev = item["dev"]["metrics"]
    return (
        cv["macro_f1"]["mean"],
        cv["top1_accuracy"]["mean"],
        dev["macro_f1"],
        dev["top1_accuracy"],
    )


def candidate_grid() -> list[dict]:
    layer_sets = [
        (6,),
        (5, 6),
        (6, 7),
        (5, 6, 7),
        (5, 6, 8),
        (6, 7, 8),
        (4, 5, 6, 7, 8),
        (1, 5, 6, 7),
    ]
    transforms = ["raw_l2", "standard_l2", "pca64"]
    candidates = []
    for layers in layer_sets:
        for transform in transforms:
            if transform == "raw_l2" and len(layers) > 5:
                continue
            for k in (1, 3, 5, 9, 15):
                candidates.append({
                    "method": "knn",
                    "layers": list(layers),
                    "transform": transform,
                    "k": k,
                    "weighted": False,
                    "temperature": 0.1,
                })
                for temperature in (0.05, 0.1, 0.2):
                    candidates.append({
                        "method": "knn",
                        "layers": list(layers),
                        "transform": transform,
                        "k": k,
                        "weighted": True,
                        "temperature": temperature,
                    })
            for temperature in (0.05, 0.1, 0.2):
                candidates.append({
                    "method": "soft_vote",
                    "layers": list(layers),
                    "transform": transform,
                    "temperature": temperature,
                })
            for scale in (8.0, 16.0, 32.0):
                candidates.append({
                    "method": "centroid",
                    "layers": list(layers),
                    "transform": transform,
                    "scale": scale,
                })
    return candidates


def aggregate_cv(items: list[dict]) -> dict:
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
    repeat_metrics = []
    layers = tuple(candidate["layers"])
    base_features = flatten(train, layers)
    for repeat in range(repeats):
        probabilities = np.zeros((len(train.labels), len(classes)), dtype=np.float64)
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fit_indices, validation_indices in splitter.split(
            base_features,
            train.labels,
            train.groups,
        ):
            fit_features, validation_features = fit_transform_features(
                base_features[fit_indices],
                train.labels[fit_indices],
                base_features[validation_indices],
                candidate["transform"],
            )
            probabilities[validation_indices] = predict_similarity(
                fit_features,
                train.labels[fit_indices],
                validation_features,
                classes,
                candidate,
            )
        repeat_metrics.append(metrics(probabilities, train.labels, classes))
    return {
        "aggregate": aggregate_cv(repeat_metrics),
        "repeat_metrics": repeat_metrics,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(path: Path, evaluation: dict) -> None:
    selected = evaluation["selected_candidate"]
    lines = [
        "# P4 Similarity Search",
        "",
        "Protocol: no new data, no producer-specific rules, no final-based tuning. Dev ranks candidates; train-only grouped CV picks the final candidate.",
        "",
        "## Selected Candidate",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| method | `{selected['candidate']['method']}` |",
        f"| layers | `{selected['candidate']['layers']}` |",
        f"| transform | `{selected['candidate']['transform']}` |",
        f"| params | `{json.dumps(selected['candidate'], sort_keys=True)}` |",
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
    for name, passed in evaluation["success_gates"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.extend([
        "",
        "## Top CV-Checked Candidates",
        "",
        "| Rank | Method | Layers | Transform | CV Top-1 | CV Macro-F1 | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(evaluation["cv_checked"], start=1):
        candidate = item["candidate"]
        lines.append(
            f"| {index} | `{candidate['method']}` | `{candidate['layers']}` | "
            f"`{candidate['transform']}` | "
            f"{pct(item['cv']['aggregate']['top1_accuracy']['mean'])} | "
            f"{pct(item['cv']['aggregate']['macro_f1']['mean'])} | "
            f"{pct(item['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(item['dev']['metrics']['macro_f1'])} | "
            f"{pct(item['final']['metrics']['top1_accuracy'])} | "
            f"{pct(item['final']['metrics']['top3_accuracy'])} | "
            f"{pct(item['final']['metrics']['macro_f1'])} |"
        )
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python scripts/30_run_p4_similarity_search.py",
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
        default=root / "data/processed/evaluations/p4_similarity_search.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_SIMILARITY_SEARCH.md",
    )
    parser.add_argument(
        "--strategy",
        default="source_clean",
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--top-dev", type=int, default=40)
    parser.add_argument("--top-cv", type=int, default=12)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("loading features", flush=True)
    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    classes = np.unique(train.labels)
    audit_flags = load_train_audit_flags(args.audit)
    train = apply_mask(train, filter_mask(train, audit_flags, args.strategy))

    results = []
    candidates = candidate_grid()
    print(f"candidates={len(candidates)}", flush=True)
    feature_cache = {}
    for index, candidate in enumerate(candidates, start=1):
        if index % 200 == 0:
            print(f"candidate={index}/{len(candidates)}", flush=True)
        try:
            key = (tuple(candidate["layers"]), candidate["transform"])
            if key not in feature_cache:
                layers = tuple(candidate["layers"])
                feature_cache[key] = transform_feature_triplet(
                    flatten(train, layers),
                    train.labels,
                    flatten(dev, layers),
                    flatten(final, layers),
                    candidate["transform"],
                )
            train_features, dev_features, final_features = feature_cache[key]
            dev_probabilities = predict_similarity(
                train_features,
                train.labels,
                dev_features,
                classes,
                candidate,
            )
            final_probabilities = predict_similarity(
                train_features,
                train.labels,
                final_features,
                classes,
                candidate,
            )
            result = {
                "candidate": candidate,
                "dev": {"metrics": metrics(dev_probabilities, dev.labels, classes)},
                "final": {"metrics": metrics(final_probabilities, final.labels, classes)},
            }
        except Exception as exc:  # noqa: BLE001
            result = {"candidate": candidate, "failed": True, "error": str(exc)}
        results.append(result)

    successful = [item for item in results if not item.get("failed")]
    top_dev = sorted(successful, key=selection_key, reverse=True)[:args.top_dev]
    cv_checked = []
    for index, item in enumerate(top_dev[:args.top_cv], start=1):
        candidate = item["candidate"]
        print(
            f"cv {index}/{args.top_cv}: {candidate['method']} "
            f"{candidate['layers']} {candidate['transform']}",
            flush=True,
        )
        cv = cross_validate_candidate(
            candidate,
            train,
            classes,
            repeats=args.cv_repeats,
            folds=args.cv_folds,
            seed=args.seed,
        )
        cv_checked.append({**item, "cv": cv})

    selected = max(cv_checked, key=cv_selection_key)
    gates = success_gates(selected["final"]["metrics"])
    evaluation = {
        "protocol": {
            "purpose": "P4 global similarity search",
            "strategy": args.strategy,
            "selection": (
                "Dev ranks candidates; top candidates receive train-only grouped CV. "
                "Final is report-only after selection."
            ),
            "cv": {
                "repeats": args.cv_repeats,
                "folds": args.cv_folds,
                "seed": args.seed,
            },
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "minimum_class_songs": int(min(Counter(train.labels).values())),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
        },
        "top_dev_candidates": top_dev,
        "cv_checked": cv_checked,
        "selected_candidate": selected,
        "success_gates": gates,
        "target_met": bool(any(gates.values())),
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        "selected": selected["candidate"],
        "cv": selected["cv"]["aggregate"],
        "dev": selected["dev"]["metrics"],
        "final": selected["final"]["metrics"],
        "success_gates": gates,
        "target_met": evaluation["target_met"],
        "output": str(args.output),
        "report": str(args.report_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
