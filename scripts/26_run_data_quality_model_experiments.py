#!/usr/bin/env python
"""Compare data-quality filters and global classifier heads.

This script is intentionally experimental: it never deletes cached data and it
does not write producer-specific rules.  It compares reproducible training-song
filters plus global model-head variants against the existing dev/final splits.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibration import confidence_signals, select_rejection_threshold
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.utils.paths import project_root


DEFAULT_LAYER_SET = (5, 6, 8)
METRIC_NAMES = ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr")


@dataclass(frozen=True)
class SplitFeatures:
    layers: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    song_ids: list[str]
    titles: list[str]


@dataclass(frozen=True)
class AuditFlag:
    song_id: str
    producer_slug: str
    title: str
    codes: tuple[str, ...]
    risk_score: int


def load_split(path: Path) -> SplitFeatures:
    records = load_embedding_manifest(path)
    layers, metadata = build_song_layer_feature_matrix(records, mode="mean")
    return SplitFeatures(
        layers=layers,
        labels=np.asarray([item.producer_slug for item in metadata]),
        groups=np.asarray([item.work_id for item in metadata]),
        song_ids=[item.song_id for item in metadata],
        titles=[item.title for item in metadata],
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
    flags: dict[str, AuditFlag] = {}
    for record in audit["records"]:
        if record.get("split") != "train" or not record.get("flags"):
            continue
        song_id = source_key(record)
        if not song_id:
            continue
        flags[song_id] = AuditFlag(
            song_id=song_id,
            producer_slug=record["producer_slug"],
            title=record["title"],
            codes=tuple(flag["code"] for flag in record["flags"]),
            risk_score=int(record.get("risk_score") or 0),
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
        "protocol_complete": {
            "configured_source_not_original",
            "low_vocadb_rating",
            "review_pv_author",
            "missing_vocadb_song_id",
        },
    }
    if strategy not in codes_by_strategy:
        raise ValueError(f"Unknown filter strategy: {strategy}")
    blocked_codes = codes_by_strategy[strategy]
    keep = []
    for song_id in split.song_ids:
        flag = audit_flags.get(song_id)
        excluded = flag is not None and bool(set(flag.codes) & blocked_codes)
        keep.append(not excluded)
    return np.asarray(keep, dtype=bool)


def apply_mask(split: SplitFeatures, mask: np.ndarray) -> SplitFeatures:
    return SplitFeatures(
        layers=split.layers[mask],
        labels=split.labels[mask],
        groups=split.groups[mask],
        song_ids=[song_id for song_id, keep in zip(split.song_ids, mask) if keep],
        titles=[title for title, keep in zip(split.titles, mask) if keep],
    )


def true_indices(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {str(label): i for i, label in enumerate(classes)}
    return np.asarray([index[str(label)] for label in labels], dtype=int)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def ranking_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    probabilities = normalize_probabilities(probabilities)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = classes[order[:, 0]]
    lookup = {str(label): i for i, label in enumerate(classes)}
    ranks = np.asarray([
        int(np.where(order[row] == lookup[str(label)])[0][0]) + 1
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
            "log_loss": float(log_loss(
                true_indices(labels, classes),
                np.clip(probabilities, 1e-12, 1.0),
                labels=np.arange(len(classes)),
            )),
        },
        predictions,
        ranks,
    )


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    threshold: float | None = None,
) -> dict:
    probabilities = normalize_probabilities(probabilities)
    metrics, predictions, ranks = ranking_metrics(probabilities, labels, classes)
    output = {
        "metrics": metrics,
        "wrong_count": int(np.sum(predictions != labels)),
        "top3_miss_count": int(np.sum(ranks > 3)),
    }
    if threshold is not None:
        signals = confidence_signals(probabilities)
        accepted = signals["confidence"] >= threshold
        correct = predictions == labels
        output["rejection"] = {
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
    return output


def method_record(
    name: str,
    extra: dict,
    dev_probabilities: np.ndarray,
    final_probabilities: np.ndarray,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
) -> dict:
    dev_probabilities = normalize_probabilities(dev_probabilities)
    final_probabilities = normalize_probabilities(final_probabilities)
    threshold_info = select_rejection_threshold(
        dev_probabilities,
        true_indices(dev.labels, classes),
        target_precision=0.96,
        minimum_coverage=0.1,
    )
    threshold = threshold_info["threshold"]
    return {
        "method": name,
        "extra": extra,
        "dev_rejection_selection": threshold_info,
        "dev": evaluate_probabilities(
            dev_probabilities,
            dev.labels,
            classes,
            threshold=threshold,
        ),
        "final": evaluate_probabilities(
            final_probabilities,
            final.labels,
            classes,
            threshold=threshold,
        ),
    }


def class_counts(split: SplitFeatures) -> dict[str, int]:
    return dict(sorted(Counter(split.labels).items()))


def fit_layer_lda_probabilities(
    train: SplitFeatures,
    target: SplitFeatures,
    layer_indices: tuple[int, ...],
    classes: np.ndarray,
) -> np.ndarray:
    probabilities = []
    for layer in layer_indices:
        model = SongMeanShrinkageLDA.fit(train.layers[:, layer, :], train.labels)
        layer_probabilities = model.predict_proba(target.layers[:, layer, :])
        aligned = np.zeros((len(target.labels), len(classes)), dtype=np.float64)
        for source_index, label in enumerate(model.classes_):
            target_index = int(np.where(classes == label)[0][0])
            aligned[:, target_index] = layer_probabilities[:, source_index]
        probabilities.append(aligned)
    return np.stack(probabilities)


def fuse(probabilities_by_layer: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return np.tensordot(weights, probabilities_by_layer, axes=(0, 0))


def optimize_weights(
    probabilities_by_layer: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    target = true_indices(labels, classes)
    n_layers = probabilities_by_layer.shape[0]

    def objective(weights: np.ndarray) -> float:
        mixed = fuse(probabilities_by_layer, weights)
        selected = mixed[np.arange(len(target)), target]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize(
        objective,
        x0=np.full(n_layers, 1.0 / n_layers),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_layers,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success:
        return np.full(n_layers, 1.0 / n_layers)
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def centroid_probabilities(
    train: SplitFeatures,
    target: SplitFeatures,
    layer_indices: tuple[int, ...],
    classes: np.ndarray,
    scale: float,
) -> np.ndarray:
    layer_outputs = []
    for layer in layer_indices:
        centroids = []
        for label in classes:
            class_features = train.layers[train.labels == label, layer, :]
            centroid = class_features.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm > 0 else centroid)
        centroids = np.stack(centroids)
        features = target.layers[:, layer, :]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
        layer_outputs.append(softmax(scale * (features @ centroids.T), axis=1))
    return np.mean(np.stack(layer_outputs), axis=0)


def select_centroid_scale(
    train: SplitFeatures,
    dev: SplitFeatures,
    layer_indices: tuple[int, ...],
    classes: np.ndarray,
) -> tuple[float, dict]:
    candidates = (2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
    scored = []
    for scale in candidates:
        probabilities = centroid_probabilities(train, dev, layer_indices, classes, scale)
        scored.append((scale, evaluate_probabilities(probabilities, dev.labels, classes)))
    return max(
        scored,
        key=lambda item: (
            item[1]["metrics"]["macro_f1"],
            item[1]["metrics"]["top1_accuracy"],
            item[1]["metrics"]["top3_accuracy"],
            -item[1]["metrics"]["log_loss"],
        ),
    )


def select_layers_by_dev(
    train: SplitFeatures,
    dev: SplitFeatures,
    classes: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    scored = []
    for layer in range(train.layers.shape[1]):
        probabilities = fit_layer_lda_probabilities(train, dev, (layer,), classes)[0]
        scored.append((layer, evaluate_probabilities(probabilities, dev.labels, classes)))
    return tuple(sorted(
        layer for layer, _ in sorted(
            scored,
            key=lambda item: (
                item[1]["metrics"]["macro_f1"],
                item[1]["metrics"]["top1_accuracy"],
                item[1]["metrics"]["top3_accuracy"],
                -item[1]["metrics"]["log_loss"],
            ),
            reverse=True,
        )[:count]
    ))


def evaluate_method(
    name: str,
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    predict: Callable[[SplitFeatures], np.ndarray],
    extra: dict | None = None,
) -> dict:
    dev_probabilities = predict(dev)
    final_probabilities = predict(final)
    return method_record(
        name,
        extra or {},
        dev_probabilities,
        final_probabilities,
        dev,
        final,
        classes,
    )


def run_for_filter(
    strategy: str,
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
) -> dict:
    counts = class_counts(train)
    minimum_class_songs = min(counts.values())
    if set(train.labels) != set(classes):
        raise ValueError(f"Filter {strategy} dropped one or more classes")

    methods = []

    default_layer_probabilities = {
        "dev": fit_layer_lda_probabilities(train, dev, DEFAULT_LAYER_SET, classes),
        "final": fit_layer_lda_probabilities(train, final, DEFAULT_LAYER_SET, classes),
    }
    default_weights = np.full(len(DEFAULT_LAYER_SET), 1.0 / len(DEFAULT_LAYER_SET))
    methods.append(method_record(
        "lda_equal_568",
        {"layers": list(DEFAULT_LAYER_SET), "weights": default_weights.tolist()},
        fuse(default_layer_probabilities["dev"], default_weights),
        fuse(default_layer_probabilities["final"], default_weights),
        dev,
        final,
        classes,
    ))

    selected_layers = select_layers_by_dev(train, dev, classes, 3)
    selected_dev_layers = fit_layer_lda_probabilities(train, dev, selected_layers, classes)
    selected_final_layers = fit_layer_lda_probabilities(train, final, selected_layers, classes)
    selected_equal = np.full(len(selected_layers), 1.0 / len(selected_layers))
    methods.append(method_record(
        "lda_equal_top3_dev_layers",
        {"layers": list(selected_layers), "weights": selected_equal.tolist()},
        fuse(selected_dev_layers, selected_equal),
        fuse(selected_final_layers, selected_equal),
        dev,
        final,
        classes,
    ))

    all_layers = tuple(range(train.layers.shape[1]))
    all_dev_layers = fit_layer_lda_probabilities(train, dev, all_layers, classes)
    all_final_layers = fit_layer_lda_probabilities(train, final, all_layers, classes)
    all_weights = optimize_weights(all_dev_layers, dev.labels, classes)
    methods.append(method_record(
        "lda_dev_weighted_all_layers",
        {
            "layers": list(all_layers),
            "weights": [float(weight) for weight in all_weights],
        },
        fuse(all_dev_layers, all_weights),
        fuse(all_final_layers, all_weights),
        dev,
        final,
        classes,
    ))

    centroid_scale, centroid_dev = select_centroid_scale(
        train,
        dev,
        selected_layers,
        classes,
    )
    methods.append(method_record(
        "centroid_equal_top3_dev_layers",
        {"layers": list(selected_layers), "scale": centroid_scale},
        centroid_probabilities(train, dev, selected_layers, classes, centroid_scale),
        centroid_probabilities(train, final, selected_layers, classes, centroid_scale),
        dev,
        final,
        classes,
    ))

    lda_dev = fuse(selected_dev_layers, selected_equal)
    lda_final = fuse(selected_final_layers, selected_equal)
    centroid_dev_probabilities = centroid_probabilities(
        train,
        dev,
        selected_layers,
        classes,
        centroid_scale,
    )
    centroid_final_probabilities = centroid_probabilities(
        train,
        final,
        selected_layers,
        classes,
        centroid_scale,
    )
    blend_candidates = []
    for blend in np.linspace(0.0, 1.0, 11):
        probabilities = blend * lda_dev + (1.0 - blend) * centroid_dev_probabilities
        blend_candidates.append((
            float(blend),
            evaluate_probabilities(probabilities, dev.labels, classes),
        ))
    best_blend, blend_dev = max(
        blend_candidates,
        key=lambda item: (
            item[1]["metrics"]["macro_f1"],
            item[1]["metrics"]["top1_accuracy"],
            item[1]["metrics"]["top3_accuracy"],
            -item[1]["metrics"]["log_loss"],
        ),
    )
    blend_final_probabilities = (
        best_blend * lda_final
        + (1.0 - best_blend) * centroid_final_probabilities
    )
    methods.append(method_record(
        "blend_lda_centroid_top3_dev_layers",
        {
            "layers": list(selected_layers),
            "centroid_scale": centroid_scale,
            "lda_weight": best_blend,
        },
        best_blend * lda_dev + (1.0 - best_blend) * centroid_dev_probabilities,
        blend_final_probabilities,
        dev,
        final,
        classes,
    ))

    selected = max(
        methods,
        key=lambda item: (
            item["dev"]["metrics"]["macro_f1"],
            item["dev"]["metrics"]["top1_accuracy"],
            item["dev"]["metrics"]["top3_accuracy"],
            -item["dev"]["metrics"]["log_loss"],
        ),
    )
    return {
        "strategy": strategy,
        "training_songs": int(len(train.labels)),
        "minimum_class_songs": int(minimum_class_songs),
        "songs_per_class": counts,
        "methods": methods,
        "selected_by_dev": {
            "method": selected["method"],
            "extra": selected["extra"],
            "dev": selected["dev"],
            "final": selected["final"],
        },
    }


def summarize_exclusions(
    full_train: SplitFeatures,
    filtered_train: SplitFeatures,
    audit_flags: dict[str, AuditFlag],
) -> dict:
    excluded = sorted(set(full_train.song_ids) - set(filtered_train.song_ids))
    by_code: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    records = []
    for song_id in excluded:
        flag = audit_flags.get(song_id)
        if not flag:
            continue
        by_code.update(flag.codes)
        by_class[flag.producer_slug] += 1
        records.append({
            "song_id": song_id,
            "producer_slug": flag.producer_slug,
            "title": flag.title,
            "codes": list(flag.codes),
            "risk_score": flag.risk_score,
        })
    return {
        "excluded_songs": len(excluded),
        "by_code": dict(sorted(by_code.items())),
        "by_class": dict(sorted(by_class.items())),
        "records": records,
    }


def write_filtered_manifest(
    source_manifest: Path,
    output_root: Path,
    strategy: str,
    keep_song_ids: set[str],
    exclusion_summary: dict,
) -> dict:
    output_dir = output_root / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_count = 0
    song_producers: dict[str, str] = {}
    output_manifest = output_dir / "segments.jsonl"
    with source_manifest.open("r", encoding="utf-8") as source, output_manifest.open(
        "w",
        encoding="utf-8",
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["song_id"] not in keep_song_ids:
                continue
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            segment_count += 1
            song_producers[record["song_id"]] = record["producer_slug"]

    songs_per_class = Counter(song_producers.values())
    summary = {
        "strategy": strategy,
        "source_manifest": str(source_manifest),
        "manifest": str(output_manifest),
        "songs": len(song_producers),
        "segments": segment_count,
        "classes": len(songs_per_class),
        "minimum_class_songs": min(songs_per_class.values()) if songs_per_class else 0,
        "songs_per_class": dict(sorted(songs_per_class.items())),
        "exclusions": exclusion_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "strategy": strategy,
        "manifest": str(output_manifest),
        "summary": str(output_dir / "summary.json"),
        "songs": summary["songs"],
        "segments": summary["segments"],
        "minimum_class_songs": summary["minimum_class_songs"],
    }


def write_report(path: Path, evaluation: dict) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "# P4 数据质量与全局模型头实验",
        "",
        "本实验不新增歌曲、不删除缓存文件，也不写任何特定 P 主规则。所有候选方法只使用现有训练、dev 和 final frozen 分区。",
        "",
        "## 结果摘要",
        "",
        "| 训练过滤 | Dev 选择方法 | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 错误 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in evaluation["results"]:
        selected = result["selected_by_dev"]
        lines.append(
            "| "
            f"{result['strategy']} | "
            f"{selected['method']} | "
            f"{pct(selected['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(selected['dev']['metrics']['macro_f1'])} | "
            f"{pct(selected['final']['metrics']['top1_accuracy'])} | "
            f"{pct(selected['final']['metrics']['top3_accuracy'])} | "
            f"{pct(selected['final']['metrics']['macro_f1'])} | "
            f"{selected['final']['wrong_count']} |"
        )

    lines.extend([
        "",
        "## 数据过滤",
        "",
        "`source_clean` 只排除 VocaDB 明确标为非 Original/Reprint 风险的训练歌；"
        "`review_clean` 额外排除低评分或 PV 作者需复核的训练歌；"
        "`protocol_complete` 再排除配置里缺 `vocadb_song_id` 的训练歌，仅用于衡量严格协议的样本代价。",
        "",
        "| 训练过滤 | 保留训练歌 | 最小类样本 | 排除歌曲 |",
        "|---|---:|---:|---:|",
    ])
    for result in evaluation["results"]:
        excluded = evaluation["exclusions"][result["strategy"]]["excluded_songs"]
        lines.append(
            f"| {result['strategy']} | {result['training_songs']} | "
            f"{result['minimum_class_songs']} | {excluded} |"
        )

    if evaluation.get("filtered_manifests"):
        lines.extend([
            "",
            "## 过滤后 Manifest",
            "",
            "| 训练过滤 | Manifest | 歌曲 | 分段 | 最小类样本 |",
            "|---|---|---:|---:|---:|",
        ])
        for item in evaluation["filtered_manifests"]:
            lines.append(
                f"| {item['strategy']} | `{item['manifest']}` | "
                f"{item['songs']} | {item['segments']} | "
                f"{item['minimum_class_songs']} |"
            )

    lines.extend([
        "",
        "## 候选方法",
        "",
        "| 训练过滤 | 方法 | Dev Top-1 | Dev Top-3 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 接受准确率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in evaluation["results"]:
        for method in result["methods"]:
            final_rejection = method["final"].get("rejection", {})
            accepted_accuracy = final_rejection.get("accepted_accuracy")
            accepted_text = pct(accepted_accuracy) if accepted_accuracy is not None else "-"
            lines.append(
                "| "
                f"{result['strategy']} | "
                f"{method['method']} | "
                f"{pct(method['dev']['metrics']['top1_accuracy'])} | "
                f"{pct(method['dev']['metrics']['top3_accuracy'])} | "
                f"{pct(method['dev']['metrics']['macro_f1'])} | "
                f"{pct(method['final']['metrics']['top1_accuracy'])} | "
                f"{pct(method['final']['metrics']['top3_accuracy'])} | "
                f"{pct(method['final']['metrics']['macro_f1'])} | "
                f"{accepted_text} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default=root / "data/processed/evaluations/p4_data_quality_model_experiments.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_DATA_QUALITY_MODEL_EXPERIMENTS.md",
    )
    parser.add_argument(
        "--filtered-output-root",
        type=Path,
        default=root / "data/processed/curated/p4_data_quality",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["raw", "source_clean", "review_clean", "protocol_complete"],
        choices=["raw", "source_clean", "review_clean", "protocol_complete"],
    )
    args = parser.parse_args()

    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    classes = np.unique(train.labels)
    if set(dev.labels) - set(classes) or set(final.labels) - set(classes):
        raise ValueError("Dev/final contain labels that are absent from training")

    audit_flags = load_train_audit_flags(args.audit)
    results = []
    exclusions = {}
    filtered_manifests = []
    for strategy in args.filters:
        mask = filter_mask(train, audit_flags, strategy)
        filtered_train = apply_mask(train, mask)
        exclusions[strategy] = summarize_exclusions(train, filtered_train, audit_flags)
        if strategy != "raw":
            filtered_manifests.append(write_filtered_manifest(
                args.train_manifest,
                args.filtered_output_root,
                strategy,
                set(filtered_train.song_ids),
                exclusions[strategy],
            ))
        results.append(run_for_filter(strategy, filtered_train, dev, final, classes))

    evaluation = {
        "protocol": {
            "purpose": "Data-quality filters and global model-head experiments",
            "constraints": [
                "No new songs are added.",
                "No cached audio or embedding files are deleted.",
                "No producer-specific rules or pair-specific heads are used.",
                "Development holdout is used for method selection; final frozen is reported only for acceptance.",
            ],
            "train_manifest": str(args.train_manifest),
            "dev_manifest": str(args.dev_manifest),
            "final_manifest": str(args.final_manifest),
            "audit": str(args.audit),
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
            "layers": int(train.layers.shape[1]),
            "embedding_dim": int(train.layers.shape[2]),
        },
        "exclusions": exclusions,
        "filtered_manifests": filtered_manifests,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report_output),
        "selected": {
            result["strategy"]: {
                "method": result["selected_by_dev"]["method"],
                "dev": result["selected_by_dev"]["dev"]["metrics"],
                "final": result["selected_by_dev"]["final"]["metrics"],
                "final_wrong_count": result["selected_by_dev"]["final"]["wrong_count"],
            }
            for result in results
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
