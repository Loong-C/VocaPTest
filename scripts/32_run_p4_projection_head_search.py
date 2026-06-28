#!/usr/bin/env python
"""Search regularized projection heads without adding data or final tuning."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocaptest.data.curation import load_embedding_manifest  # noqa: E402
from vocaptest.features.layer_features import build_song_layer_feature_matrix  # noqa: E402
from vocaptest.models.projection_head import (  # noqa: E402
    fit_projection_head,
    predict_projection_head,
)
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


def encode_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {str(label): index for index, label in enumerate(classes)}
    return np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)


def normalize(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def metrics(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    probabilities = normalize(probabilities)
    true_indices = encode_labels(labels, classes)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = order[:, 0]
    ranks = np.asarray([
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
        "log_loss": float(log_loss(
            true_indices,
            probabilities,
            labels=np.arange(len(classes)),
        )),
    }


def flatten(split: SplitFeatures, layers: tuple[int, ...]) -> np.ndarray:
    return split.layers[:, list(layers), :].reshape(len(split.labels), -1)


def candidate_grid() -> list[dict]:
    candidates = []
    for layers in [(6,), (7,), (5, 6, 7), (6, 7, 8), (4, 5, 6, 7, 8)]:
        for projection_dim in [64, 128]:
            for contrastive_weight in [0.0, 0.05, 0.1]:
                candidates.append({
                    "layers": list(layers),
                    "projection_dim": projection_dim,
                    "contrastive_weight": contrastive_weight,
                    "weight_decay": 0.003,
                    "learning_rate": 0.001,
                })
    return candidates


def average_projection_predictions(
    train_features: np.ndarray,
    train_label_ids: np.ndarray,
    train_groups: np.ndarray,
    target_features: np.ndarray,
    candidate: dict,
    *,
    class_count: int,
    repeats: int,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, list[int]]:
    outputs = []
    selected_epochs = []
    for repeat in range(repeats):
        model, scaler, best_epoch = fit_projection_head(
            train_features,
            train_label_ids,
            train_groups,
            class_count=class_count,
            seed=seed + repeat * 997,
            device=device,
            projection_dim=int(candidate["projection_dim"]),
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            contrastive_weight=float(candidate["contrastive_weight"]),
        )
        selected_epochs.append(int(best_epoch))
        outputs.append(predict_projection_head(model, scaler, target_features, device))
    return normalize(np.mean(np.stack(outputs), axis=0)), selected_epochs


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
        "# P4 Projection Head Search",
        "",
        "Protocol: frozen MERT song features, source-clean train set, small regularized projection heads selected on dev. Final is evaluated only for the selected dev candidate.",
        "",
        "## Selected Candidate",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| candidate | `{json.dumps(selected['candidate'], sort_keys=True)}` |",
        f"| Dev Top-1 | {pct(selected['dev']['metrics']['top1_accuracy'])} |",
        f"| Dev Top-3 | {pct(selected['dev']['metrics']['top3_accuracy'])} |",
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
        "## Top Dev Candidates",
        "",
        "| Rank | Candidate | Dev Top-1 | Dev Top-3 | Dev Macro-F1 | Mean Epochs |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(evaluation["candidate_results"][:15], start=1):
        lines.append(
            f"| {index} | `{json.dumps(item['candidate'], sort_keys=True)}` | "
            f"{pct(item['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(item['dev']['metrics']['top3_accuracy'])} | "
            f"{pct(item['dev']['metrics']['macro_f1'])} | "
            f"{np.mean(item['selected_epochs']):.1f} |"
        )
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python scripts/32_run_p4_projection_head_search.py",
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
        default=root / "data/processed/evaluations/p4_projection_head_search.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_PROJECTION_HEAD_SEARCH.md",
    )
    parser.add_argument(
        "--strategy",
        default="source_clean",
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading features", flush=True)
    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    audit_flags = load_train_audit_flags(args.audit)
    train = apply_mask(train, filter_mask(train, audit_flags, args.strategy))
    classes = np.unique(train.labels)
    train_label_ids = encode_labels(train.labels, classes)

    candidate_results = []
    for index, candidate in enumerate(candidate_grid(), start=1):
        layers = tuple(candidate["layers"])
        print(f"candidate {index}/{len(candidate_grid())}: {candidate}", flush=True)
        train_features = flatten(train, layers)
        dev_features = flatten(dev, layers)
        dev_probabilities, selected_epochs = average_projection_predictions(
            train_features,
            train_label_ids,
            train.groups,
            dev_features,
            candidate,
            class_count=len(classes),
            repeats=args.repeats,
            seed=args.seed + index * 10000,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        candidate_results.append({
            "candidate": candidate,
            "dev": {"metrics": metrics(dev_probabilities, dev.labels, classes)},
            "selected_epochs": selected_epochs,
        })

    candidate_results = sorted(candidate_results, key=selection_key, reverse=True)
    selected = candidate_results[0]
    selected_layers = tuple(selected["candidate"]["layers"])
    final_probabilities, final_epochs = average_projection_predictions(
        flatten(train, selected_layers),
        train_label_ids,
        train.groups,
        flatten(final, selected_layers),
        selected["candidate"],
        class_count=len(classes),
        repeats=args.repeats,
        seed=args.seed + 999000,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    selected = {
        **selected,
        "final": {"metrics": metrics(final_probabilities, final.labels, classes)},
        "final_selected_epochs": final_epochs,
    }
    gates = success_gates(selected["final"]["metrics"])
    evaluation = {
        "protocol": {
            "purpose": "P4 projection-head search",
            "strategy": args.strategy,
            "repeats": args.repeats,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "device": device,
            "seed": args.seed,
            "selection": (
                "Candidates are selected by dev macro-F1/top1/top3/log-loss. "
                "Final is evaluated only for the selected candidate."
            ),
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "minimum_class_songs": int(min(Counter(train.labels).values())),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
        },
        "candidate_results": candidate_results,
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
