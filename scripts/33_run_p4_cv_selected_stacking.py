#!/usr/bin/env python
"""Select stacking meta models by train-only CV, with dev as a guardrail."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
STACKING_PATH = ROOT / "scripts" / "31_run_p4_stacking_search.py"
spec = importlib.util.spec_from_file_location("p4_stacking", STACKING_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {STACKING_PATH}")
p4_stacking = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p4_stacking
spec.loader.exec_module(p4_stacking)


def meta_cv_metrics(
    train_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    candidate: dict,
    *,
    folds: int,
    seed: int,
) -> dict:
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
        probabilities[validation_indices] = p4_stacking.fit_meta_predict(
            train_probabilities[fit_indices],
            labels[fit_indices],
            train_probabilities[validation_indices],
            classes,
            candidate,
        )
    return p4_stacking.metrics(probabilities, labels, classes)


def passes_dev_guard(item: dict) -> bool:
    dev = item["dev"]["metrics"]
    return (
        dev["top1_accuracy"] >= 0.80
        and dev["top3_accuracy"] >= 0.84
        and dev["macro_f1"] >= 0.75
    )


def selection_key(item: dict) -> tuple[bool, float, float, float, float, float]:
    cv = item["cv"]["metrics"]
    dev = item["dev"]["metrics"]
    return (
        passes_dev_guard(item),
        cv["macro_f1"],
        cv["top1_accuracy"],
        cv["top3_accuracy"],
        dev["macro_f1"],
        -dev["log_loss"],
    )


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(path: Path, evaluation: dict) -> None:
    selected = evaluation["selected_candidate"]
    lines = [
        "# P4 CV-Selected Stacking",
        "",
        "Protocol: base heads create train OOF probabilities. Meta candidates are selected by train-only grouped CV, with a dev guard. Final is evaluated only for the selected candidate.",
        "",
        "## Selected Candidate",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| meta | `{json.dumps(selected['candidate'], sort_keys=True)}` |",
        f"| CV Top-1 | {pct(selected['cv']['metrics']['top1_accuracy'])} |",
        f"| CV Top-3 | {pct(selected['cv']['metrics']['top3_accuracy'])} |",
        f"| CV Macro-F1 | {pct(selected['cv']['metrics']['macro_f1'])} |",
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
        "## Top CV Candidates",
        "",
        "| Rank | Candidate | Guard | CV Top-1 | CV Top-3 | CV Macro-F1 | Dev Top-1 | Dev Top-3 | Dev Macro-F1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(evaluation["meta_results"][:15], start=1):
        lines.append(
            f"| {index} | `{json.dumps(item['candidate'], sort_keys=True)}` | "
            f"`{passes_dev_guard(item)}` | "
            f"{pct(item['cv']['metrics']['top1_accuracy'])} | "
            f"{pct(item['cv']['metrics']['top3_accuracy'])} | "
            f"{pct(item['cv']['metrics']['macro_f1'])} | "
            f"{pct(item['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(item['dev']['metrics']['top3_accuracy'])} | "
            f"{pct(item['dev']['metrics']['macro_f1'])} |"
        )
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python scripts/33_run_p4_cv_selected_stacking.py",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = p4_stacking.project_root()
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
        default=root / "data/processed/evaluations/p4_cv_selected_stacking.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P4_CV_SELECTED_STACKING.md",
    )
    parser.add_argument(
        "--strategy",
        default="source_clean",
        choices=["raw", "source_clean", "review_clean"],
    )
    parser.add_argument("--base-folds", type=int, default=5)
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("loading features", flush=True)
    train = p4_stacking.load_split(args.train_manifest)
    dev = p4_stacking.load_split(args.dev_manifest)
    final = p4_stacking.load_split(args.final_manifest)
    classes = np.unique(train.labels)
    audit_flags = p4_stacking.load_train_audit_flags(args.audit)
    train = p4_stacking.apply_mask(
        train,
        p4_stacking.filter_mask(train, audit_flags, args.strategy),
    )

    specs = p4_stacking.base_specs()
    train_probs, dev_probs, final_probs, base_diagnostics = (
        p4_stacking.build_oof_and_targets(
            train,
            dev,
            final,
            classes,
            specs,
            folds=args.base_folds,
            seed=args.seed,
        )
    )

    meta_results = []
    for index, candidate in enumerate(p4_stacking.candidate_grid(), start=1):
        print(f"meta {index}/{len(p4_stacking.candidate_grid())} {candidate}", flush=True)
        dev_probabilities = p4_stacking.fit_meta_predict(
            train_probs,
            train.labels,
            dev_probs,
            classes,
            candidate,
        )
        cv = meta_cv_metrics(
            train_probs,
            train.labels,
            train.groups,
            classes,
            candidate,
            folds=args.meta_folds,
            seed=args.seed + 4100,
        )
        meta_results.append({
            "candidate": candidate,
            "cv": {"metrics": cv},
            "dev": {"metrics": p4_stacking.metrics(dev_probabilities, dev.labels, classes)},
        })

    meta_results = sorted(meta_results, key=selection_key, reverse=True)
    selected = meta_results[0]
    final_probabilities = p4_stacking.fit_meta_predict(
        train_probs,
        train.labels,
        final_probs,
        classes,
        selected["candidate"],
    )
    selected = {
        **selected,
        "final": {"metrics": p4_stacking.metrics(final_probabilities, final.labels, classes)},
    }
    gates = p4_stacking.success_gates(selected["final"]["metrics"])
    evaluation = {
        "protocol": {
            "purpose": "P4 CV-selected stacking",
            "strategy": args.strategy,
            "base_folds": args.base_folds,
            "meta_folds": args.meta_folds,
            "seed": args.seed,
            "selection": (
                "Primary ranking uses train-only grouped CV on meta features. "
                "Dev is used only as a minimum guard. Final is report-only."
            ),
            "dev_guard": {
                "top1_accuracy": ">= 0.80",
                "top3_accuracy": ">= 0.84",
                "macro_f1": ">= 0.75",
            },
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "minimum_class_songs": int(min(p4_stacking.Counter(train.labels).values())),
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
        "cv": selected["cv"]["metrics"],
        "dev": selected["dev"]["metrics"],
        "final": selected["final"]["metrics"],
        "success_gates": gates,
        "target_met": evaluation["target_met"],
        "output": str(args.output),
        "report": str(args.report_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
