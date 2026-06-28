#!/usr/bin/env python
"""Stack global LDA heads with train-OOF meta features."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocaptest.data.curation import load_embedding_manifest  # noqa: E402
from vocaptest.features.layer_features import build_song_layer_feature_matrix  # noqa: E402
from vocaptest.models.song_lda import SongMeanShrinkageLDA  # noqa: E402
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


def flatten(split_or_layers, layers: tuple[int, ...]) -> np.ndarray:
    array = split_or_layers.layers if isinstance(split_or_layers, SplitFeatures) else split_or_layers
    return array[:, list(layers), :].reshape(array.shape[0], -1)


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


def predict_base(
    spec: dict,
    train_layers: np.ndarray,
    train_labels: np.ndarray,
    target_layers: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    layers = tuple(spec["layers"])
    if spec["kind"] == "layer":
        layer = layers[0]
        model = SongMeanShrinkageLDA.fit(train_layers[:, layer, :], train_labels)
        return align_probabilities(
            model.predict_proba(target_layers[:, layer, :]),
            model.classes_,
            classes,
        )
    if spec["kind"] == "layer_fusion":
        outputs = [
            predict_base(
                {"name": f"layer_{layer}", "kind": "layer", "layers": [layer]},
                train_layers,
                train_labels,
                target_layers,
                classes,
            )
            for layer in layers
        ]
        return normalize(np.mean(np.stack(outputs), axis=0))
    if spec["kind"] == "concat":
        model = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(len(classes), 1.0 / len(classes)),
        )
        model.fit(flatten(train_layers, layers), train_labels)
        return align_probabilities(
            model.predict_proba(flatten(target_layers, layers)),
            model.classes_,
            classes,
        )
    raise ValueError(f"Unknown base kind: {spec['kind']}")


def build_oof_and_targets(
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    specs: list[dict],
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    train_blocks = []
    dev_blocks = []
    final_blocks = []
    diagnostics = []
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    for index, spec in enumerate(specs, start=1):
        print(f"base {index}/{len(specs)} {spec['name']}", flush=True)
        oof = np.zeros((len(train.labels), len(classes)), dtype=np.float64)
        for fit_indices, validation_indices in splitter.split(
            train.layers,
            train.labels,
            train.groups,
        ):
            oof[validation_indices] = predict_base(
                spec,
                train.layers[fit_indices],
                train.labels[fit_indices],
                train.layers[validation_indices],
                classes,
            )
        dev_probabilities = predict_base(
            spec,
            train.layers,
            train.labels,
            dev.layers,
            classes,
        )
        final_probabilities = predict_base(
            spec,
            train.layers,
            train.labels,
            final.layers,
            classes,
        )
        train_blocks.append(oof)
        dev_blocks.append(dev_probabilities)
        final_blocks.append(final_probabilities)
        diagnostics.append({
            "spec": spec,
            "oof": metrics(oof, train.labels, classes),
            "dev": metrics(dev_probabilities, dev.labels, classes),
            "final": metrics(final_probabilities, final.labels, classes),
        })
    return (
        np.stack(train_blocks, axis=1),
        np.stack(dev_blocks, axis=1),
        np.stack(final_blocks, axis=1),
        diagnostics,
    )


def meta_features(probabilities: np.ndarray, mode: str) -> np.ndarray:
    if mode == "prob":
        return probabilities.reshape(probabilities.shape[0], -1)
    if mode == "log_prob":
        return np.log(np.clip(probabilities, 1e-12, 1.0)).reshape(probabilities.shape[0], -1)
    if mode == "prob_and_log":
        return np.concatenate([
            probabilities.reshape(probabilities.shape[0], -1),
            np.log(np.clip(probabilities, 1e-12, 1.0)).reshape(probabilities.shape[0], -1),
        ], axis=1)
    raise ValueError(f"Unknown meta mode: {mode}")


def fit_meta_predict(
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    target_probabilities: np.ndarray,
    classes: np.ndarray,
    candidate: dict,
) -> np.ndarray:
    train_features = meta_features(train_probabilities, candidate["feature_mode"])
    target_features = meta_features(target_probabilities, candidate["feature_mode"])
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    target_features = scaler.transform(target_features)
    if candidate["kind"] == "logreg":
        model = LogisticRegression(
            C=float(candidate["C"]),
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(train_features, train_labels)
        return align_probabilities(model.predict_proba(target_features), model.classes_, classes)
    if candidate["kind"] == "ridge":
        model = RidgeClassifier(
            alpha=float(candidate["alpha"]),
            class_weight="balanced",
        )
        model.fit(train_features, train_labels)
        scores = model.decision_function(target_features)
        scores = scores - scores.max(axis=1, keepdims=True)
        return align_probabilities(np.exp(scores), model.classes_, classes)
    raise ValueError(f"Unknown meta kind: {candidate['kind']}")


def candidate_grid() -> list[dict]:
    candidates = []
    for feature_mode in ("prob", "log_prob", "prob_and_log"):
        for c_value in (0.03, 0.1, 0.3, 1.0, 3.0):
            candidates.append({
                "kind": "logreg",
                "feature_mode": feature_mode,
                "C": c_value,
            })
        for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
            candidates.append({
                "kind": "ridge",
                "feature_mode": feature_mode,
                "alpha": alpha,
            })
    return candidates


def selection_key(item: dict) -> tuple[float, float, float, float]:
    dev = item["dev"]["metrics"]
    return (
        dev["macro_f1"],
        dev["top1_accuracy"],
        dev["top3_accuracy"],
        -dev["log_loss"],
    )


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(path: Path, evaluation: dict) -> None:
    selected = evaluation["selected_candidate"]
    lines = [
        "# P4 Stacking Search",
        "",
        "Protocol: base heads create train OOF probabilities; meta heads are selected on dev. Final is report-only.",
        "",
        "## Selected Candidate",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| meta | `{json.dumps(selected['candidate'], sort_keys=True)}` |",
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
        "## Top Meta Candidates",
        "",
        "| Rank | Candidate | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(evaluation["meta_results"][:15], start=1):
        lines.append(
            f"| {index} | `{json.dumps(item['candidate'], sort_keys=True)}` | "
            f"{pct(item['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(item['dev']['metrics']['macro_f1'])} | "
            f"{pct(item['final']['metrics']['top1_accuracy'])} | "
            f"{pct(item['final']['metrics']['top3_accuracy'])} | "
            f"{pct(item['final']['metrics']['macro_f1'])} |"
        )
    lines.extend([
        "",
        "## Base Heads",
        "",
        "| Base | OOF Top-1 | Dev Top-1 | Final Top-1 | Final Top-3 | Final Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in evaluation["base_diagnostics"]:
        lines.append(
            f"| `{item['spec']['name']}` | "
            f"{pct(item['oof']['top1_accuracy'])} | "
            f"{pct(item['dev']['top1_accuracy'])} | "
            f"{pct(item['final']['top1_accuracy'])} | "
            f"{pct(item['final']['top3_accuracy'])} | "
            f"{pct(item['final']['macro_f1'])} |"
        )
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python scripts/31_run_p4_stacking_search.py",
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
        default=root / "data/processed/evaluations/p4_stacking_search.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_STACKING_SEARCH.md",
    )
    parser.add_argument(
        "--strategy",
        default="source_clean",
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("loading features", flush=True)
    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    classes = np.unique(train.labels)
    audit_flags = load_train_audit_flags(args.audit)
    train = apply_mask(train, filter_mask(train, audit_flags, args.strategy))

    specs = base_specs()
    train_probs, dev_probs, final_probs, base_diagnostics = build_oof_and_targets(
        train,
        dev,
        final,
        classes,
        specs,
        folds=args.folds,
        seed=args.seed,
    )

    meta_results = []
    for candidate in candidate_grid():
        dev_probabilities = fit_meta_predict(
            train_probs,
            train.labels,
            dev_probs,
            classes,
            candidate,
        )
        final_probabilities = fit_meta_predict(
            train_probs,
            train.labels,
            final_probs,
            classes,
            candidate,
        )
        meta_results.append({
            "candidate": candidate,
            "dev": {"metrics": metrics(dev_probabilities, dev.labels, classes)},
            "final": {"metrics": metrics(final_probabilities, final.labels, classes)},
        })
    meta_results = sorted(meta_results, key=selection_key, reverse=True)
    selected = meta_results[0]
    gates = success_gates(selected["final"]["metrics"])
    evaluation = {
        "protocol": {
            "purpose": "P4 stacking global model-head search",
            "strategy": args.strategy,
            "folds": args.folds,
            "seed": args.seed,
            "selection": (
                "Base heads use train-only grouped OOF probabilities. Meta "
                "candidates are selected by dev metrics. Final is report-only."
            ),
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "minimum_class_songs": int(min(Counter(train.labels).values())),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
        },
        "base_specs": specs,
        "base_diagnostics": base_diagnostics,
        "meta_results": meta_results,
        "selected_candidate": selected,
        "success_gates": gates,
        "target_met": bool(any(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        "selected": selected["candidate"],
        "dev": selected["dev"]["metrics"],
        "final": selected["final"]["metrics"],
        "success_gates": gates,
        "target_met": evaluation["target_met"],
        "output": str(args.output),
        "report": str(args.report_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
