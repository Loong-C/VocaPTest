#!/usr/bin/env python
"""Cross-validate the strongest P4 broad-search candidates on train only."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocaptest.data.curation import load_embedding_manifest  # noqa: E402
from vocaptest.features.layer_features import build_song_layer_feature_matrix  # noqa: E402
from vocaptest.models.song_lda import SongMeanShrinkageLDA  # noqa: E402
from vocaptest.utils.paths import project_root  # noqa: E402


@dataclass(frozen=True)
class SplitFeatures:
    layers: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    song_ids: list[str]


@dataclass(frozen=True)
class AuditFlag:
    song_id: str
    producer_slug: str
    codes: tuple[str, ...]


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
    blocked_codes = codes_by_strategy[strategy]
    keep = []
    for song_id in split.song_ids:
        flag = audit_flags.get(song_id)
        keep.append(flag is None or not bool(set(flag.codes) & blocked_codes))
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


def aggregate(items: list[dict]) -> dict:
    output = {}
    for name in ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr", "log_loss"):
        values = np.asarray([item[name] for item in items], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def arithmetic_layer_lda(
    train_layers: np.ndarray,
    train_labels: np.ndarray,
    test_layers: np.ndarray,
    layers: tuple[int, ...],
    classes: np.ndarray,
) -> np.ndarray:
    outputs = []
    for layer in layers:
        model = SongMeanShrinkageLDA.fit(train_layers[:, layer, :], train_labels)
        outputs.append(align_probabilities(
            model.predict_proba(test_layers[:, layer, :]),
            model.classes_,
            classes,
        ))
    return normalize(np.mean(np.stack(outputs), axis=0))


def temperature_layer_lda(
    train_layers: np.ndarray,
    train_labels: np.ndarray,
    test_layers: np.ndarray,
    layers: tuple[int, ...],
    classes: np.ndarray,
    temperature: float,
) -> np.ndarray:
    outputs = []
    for layer in layers:
        model = SongMeanShrinkageLDA.fit(train_layers[:, layer, :], train_labels)
        probabilities = align_probabilities(
            model.predict_proba(test_layers[:, layer, :]),
            model.classes_,
            classes,
        )
        outputs.append(np.exp(np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature))
    return normalize(np.mean(np.stack(outputs), axis=0))


def concat_lda(
    train_layers: np.ndarray,
    train_labels: np.ndarray,
    test_layers: np.ndarray,
    layers: tuple[int, ...],
    classes: np.ndarray,
) -> np.ndarray:
    train_features = train_layers[:, list(layers), :].reshape(len(train_labels), -1)
    test_features = test_layers[:, list(layers), :].reshape(len(test_layers), -1)
    model = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
        priors=np.full(len(classes), 1.0 / len(classes)),
    )
    model.fit(train_features, train_labels)
    return align_probabilities(model.predict_proba(test_features), model.classes_, classes)


def evaluate_candidate(
    split: SplitFeatures,
    candidate: dict,
    *,
    repeats: int,
    folds: int,
    seed: int,
) -> dict:
    classes = np.unique(split.labels)
    repeat_metrics = []
    for repeat in range(repeats):
        probabilities = np.zeros((len(split.labels), len(classes)), dtype=np.float64)
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for train_indices, validation_indices in splitter.split(
            split.layers,
            split.labels,
            split.groups,
        ):
            kwargs = candidate["kwargs"]
            if candidate["kind"] == "arithmetic_layer_lda":
                fold_probabilities = arithmetic_layer_lda(
                    split.layers[train_indices],
                    split.labels[train_indices],
                    split.layers[validation_indices],
                    tuple(kwargs["layers"]),
                    classes,
                )
            elif candidate["kind"] == "temperature_layer_lda":
                fold_probabilities = temperature_layer_lda(
                    split.layers[train_indices],
                    split.labels[train_indices],
                    split.layers[validation_indices],
                    tuple(kwargs["layers"]),
                    classes,
                    float(kwargs["temperature"]),
                )
            elif candidate["kind"] == "concat_lda":
                fold_probabilities = concat_lda(
                    split.layers[train_indices],
                    split.labels[train_indices],
                    split.layers[validation_indices],
                    tuple(kwargs["layers"]),
                    classes,
                )
            else:
                raise ValueError(f"Unknown candidate kind: {candidate['kind']}")
            probabilities[validation_indices] = fold_probabilities
        repeat_metrics.append(metrics(probabilities, split.labels, classes))
    return {
        "candidate": candidate,
        "aggregate": aggregate(repeat_metrics),
        "repeat_metrics": repeat_metrics,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(path: Path, evaluation: dict) -> None:
    lines = [
        "# P4 Candidate Cross-Validation",
        "",
        "这里只在训练集内部做 grouped cross-validation，用来判断 broad search 里几个候选方向是否只是 dev/final 偶然波动。",
        "",
        "| 过滤 | 候选 | CV Top-1 mean | CV Top-1 std | CV Top-3 mean | CV Macro-F1 mean | CV LogLoss mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in evaluation["results"]:
        for item in result["candidates"]:
            aggregate_result = item["aggregate"]
            lines.append(
                "| "
                f"{result['strategy']} | "
                f"{item['candidate']['name']} | "
                f"{pct(aggregate_result['top1_accuracy']['mean'])} | "
                f"{pct(aggregate_result['top1_accuracy']['std'])} | "
                f"{pct(aggregate_result['top3_accuracy']['mean'])} | "
                f"{pct(aggregate_result['macro_f1']['mean'])} | "
                f"{aggregate_result['log_loss']['mean']:.3f} |"
            )
    lines.extend([
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts/28_validate_p4_broad_candidates.py",
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
        "--audit",
        type=Path,
        default=root / "data/processed/evaluations/catalog_risk_audit.json",
    )
    parser.add_argument(
        "--broad-search",
        type=Path,
        default=root / "data/processed/evaluations/p4_broad_model_search.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/processed/evaluations/p4_candidate_cross_validation.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_CANDIDATE_CROSS_VALIDATION.md",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    broad = json.loads(args.broad_search.read_text(encoding="utf-8"))
    train = load_split(args.train_manifest)
    audit_flags = load_train_audit_flags(args.audit)

    candidates_by_strategy = {
        "raw": [
            {
                "name": "baseline_lda_568",
                "kind": "arithmetic_layer_lda",
                "kwargs": {"layers": [5, 6, 8]},
            },
            {
                "name": "raw_concat_lda_mid_4_8",
                "kind": "concat_lda",
                "kwargs": {"layers": [4, 5, 6, 7, 8]},
            },
        ],
        "source_clean": [
            {
                "name": "source_clean_lda_top3_567",
                "kind": "arithmetic_layer_lda",
                "kwargs": {"layers": [5, 6, 7]},
            },
            {
                "name": "source_clean_temperature_1567",
                "kind": "temperature_layer_lda",
                "kwargs": {"layers": [1, 5, 6, 7], "temperature": 2.0},
            },
            {
                "name": "source_clean_concat_lda_567",
                "kind": "concat_lda",
                "kwargs": {"layers": [5, 6, 7]},
            },
        ],
        "review_clean": [
            {
                "name": "review_clean_lda_568",
                "kind": "arithmetic_layer_lda",
                "kwargs": {"layers": [5, 6, 8]},
            },
            {
                "name": "review_clean_temperature_236811",
                "kind": "temperature_layer_lda",
                "kwargs": {"layers": [2, 3, 6, 8, 11], "temperature": 3.0},
            },
        ],
    }

    results = []
    for strategy, candidates in candidates_by_strategy.items():
        print(f"strategy={strategy}", flush=True)
        filtered = apply_mask(train, filter_mask(train, audit_flags, strategy))
        evaluated = [
            evaluate_candidate(
                filtered,
                candidate,
                repeats=args.repeats,
                folds=args.folds,
                seed=args.seed,
            )
            for candidate in candidates
        ]
        results.append({
            "strategy": strategy,
            "training_songs": int(len(filtered.labels)),
            "minimum_class_songs": int(min(Counter(filtered.labels).values())),
            "broad_search_selected": next(
                item["selected_by_dev"]
                for item in broad["results"]
                if item["strategy"] == strategy
            ),
            "candidates": evaluated,
        })

    evaluation = {
        "protocol": {
            "purpose": "Grouped train-only validation for P4 broad-search candidates",
            "repeats": args.repeats,
            "folds": args.folds,
            "seed": args.seed,
            "selection_note": (
                "This is not final tuning; it checks whether candidates are stable "
                "under train-only grouped resampling."
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        result["strategy"]: {
            item["candidate"]["name"]: item["aggregate"]["top1_accuracy"]["mean"]
            for item in result["candidates"]
        }
        for result in results
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
