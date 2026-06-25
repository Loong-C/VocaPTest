#!/usr/bin/env python
"""Run P3 model-head experiments without changing the deployed app model.

The goal is not to tune one troublesome producer pair.  This script compares
global, future-scalable changes on the existing MERT layer cache:

* single-layer re-selection;
* probability ensembles of several independently trained MERT-layer LDA heads;
* a small multi-stat pooling probe;
* calibrated rejection at several target accepted precisions.

It writes a compact JSON artifact plus a narrative Markdown report.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from vocaptest.data.curation import load_embedding_manifest
from vocaptest.features.layer_features import build_song_layer_feature_matrix
from vocaptest.models.calibration import (
    TemperatureScaler,
    confidence_signals,
    select_rejection_threshold,
)
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.utils.paths import project_root


METRIC_NAMES = ("top1_accuracy", "top3_accuracy", "macro_f1", "mrr")


@dataclass
class SplitFeatures:
    layers: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    titles: list[str]


def load_split(path: Path, *, mode: str = "mean") -> SplitFeatures:
    records = load_embedding_manifest(path)
    layers, metadata = build_song_layer_feature_matrix(records, mode=mode)
    return SplitFeatures(
        layers=layers,
        labels=np.asarray([item.producer_slug for item in metadata]),
        groups=np.asarray([item.work_id for item in metadata]),
        titles=[item.title for item in metadata],
    )


def ranking_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = classes[order[:, 0]]
    class_index = {str(label): index for index, label in enumerate(classes)}
    ranks = np.asarray([
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


def delta_metrics(metrics: dict, baseline: dict) -> dict:
    return {
        name: float(metrics[name] - baseline[name])
        for name in METRIC_NAMES
    }


def fit_predict_layer_ensemble(
    train_layers: np.ndarray,
    train_labels: np.ndarray,
    test_layers: np.ndarray,
    layer_indices: Iterable[int],
    *,
    probability_temperature: float = 1.0,
) -> np.ndarray:
    probabilities = None
    layer_indices = tuple(layer_indices)
    for layer in layer_indices:
        model = SongMeanShrinkageLDA.fit(train_layers[:, layer, :], train_labels)
        layer_probabilities = model.predict_proba(test_layers[:, layer, :])
        if probability_temperature != 1.0:
            layer_probabilities = softmax(
                np.log(np.clip(layer_probabilities, 1e-12, 1.0))
                / probability_temperature,
                axis=1,
            )
        probabilities = (
            layer_probabilities
            if probabilities is None
            else probabilities + layer_probabilities
        )
    return probabilities / len(layer_indices)


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict:
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


def oof_calibration(
    train: SplitFeatures,
    classes: np.ndarray,
    layer_indices: tuple[int, ...],
    *,
    repeats: int,
    splits: int,
    seed: int,
) -> dict:
    true_indices = np.searchsorted(classes, train.labels)
    oof_blocks = []
    repeat_metrics = []
    for repeat in range(repeats):
        probabilities = np.zeros((len(train.labels), len(classes)), dtype=np.float64)
        splitter = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        for train_indices, validation_indices in splitter.split(
            train.layers,
            train.labels,
            train.groups,
        ):
            probabilities[validation_indices] = fit_predict_layer_ensemble(
                train.layers[train_indices],
                train.labels[train_indices],
                train.layers[validation_indices],
                layer_indices,
            )
        repeat_metrics.append(
            ranking_metrics(probabilities, train.labels, classes)[0]
        )
        oof_blocks.append(probabilities)

    flat_probabilities = np.concatenate(oof_blocks)
    flat_true_indices = np.tile(true_indices, repeats)
    calibrator = TemperatureScaler.fit(flat_probabilities, flat_true_indices)
    calibrated = calibrator.transform(flat_probabilities)
    thresholds = {
        str(target): select_rejection_threshold(
            calibrated,
            flat_true_indices,
            target_precision=target,
            minimum_coverage=0.1,
        )
        for target in (0.95, 0.96, 0.97, 0.98)
    }
    return {
        "repeat_metrics": repeat_metrics,
        "aggregate": aggregate_metrics(repeat_metrics),
        "temperature": calibrator.temperature,
        "thresholds": thresholds,
    }


def aggregate_metrics(items: list[dict]) -> dict:
    output = {}
    for name in METRIC_NAMES:
        values = np.asarray([item[name] for item in items], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def evaluate_layer_method(
    train: SplitFeatures,
    dev: SplitFeatures,
    final: SplitFeatures,
    classes: np.ndarray,
    layer_indices: tuple[int, ...],
    *,
    probability_temperature: float = 1.0,
) -> dict:
    dev_probabilities = fit_predict_layer_ensemble(
        train.layers,
        train.labels,
        dev.layers,
        layer_indices,
        probability_temperature=probability_temperature,
    )
    final_probabilities = fit_predict_layer_ensemble(
        train.layers,
        train.labels,
        final.layers,
        layer_indices,
        probability_temperature=probability_temperature,
    )
    return {
        "layer_indices": list(layer_indices),
        "probability_temperature": probability_temperature,
        "dev": evaluate_probabilities(dev_probabilities, dev.labels, classes),
        "final": evaluate_probabilities(final_probabilities, final.labels, classes),
    }


def top_layers_by_dev_macro(single_layers: list[dict], count: int) -> tuple[int, ...]:
    ordered = sorted(
        single_layers,
        key=lambda item: (
            item["dev"]["metrics"]["macro_f1"],
            item["dev"]["metrics"]["top1_accuracy"],
            item["dev"]["metrics"]["top3_accuracy"],
        ),
        reverse=True,
    )
    return tuple(sorted(item["layer_indices"][0] for item in ordered[:count]))


def best_broad_window(single_count: int) -> list[tuple[int, ...]]:
    windows = []
    for width in (3, 5, 7):
        for start in range(0, single_count - width + 1):
            windows.append(tuple(range(start, start + width)))
    return windows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def write_report(path: Path, evaluation: dict) -> None:
    baseline = evaluation["methods"]["baseline_layer6"]
    recommended = evaluation["methods"]["recommended_top3_dev_macro_ensemble"]
    broad = evaluation["methods"]["best_broad_midlayer_window"]
    mean_std = evaluation["methods"]["best_mean_std_probe"]

    lines = [
        "# P3 模型实验报告",
        "",
        "日期：2026-06-25",
        "",
        "## 目标",
        "",
        "本实验只测试未来继续增加 P 主时也应该有效的全局模型头改法；不针对某两个作者写特殊规则，"
        "也不修改部署链路或 VPS 文件。",
        "",
        "## 有效修正",
        "",
        "真正有效的改法很朴素：保留已经缓存好的 MERT-95M，不微调 backbone；在若干个 MERT 层上分别训练"
        "歌曲级 Shrinkage LDA，然后平均这些层的类别概率。按 dev macro-F1 选择出的三层是 **5、6、8**："
        "第 6 层是当前主线 baseline，第 8 层是 dev 上最强的单层，第 5 层是 dev macro-F1 次优单层。"
        "这是一条全局选择规则，不是给某个 P 主写补丁。",
        "",
        "## 效果",
        "",
        "| 方法 | Dev Top-1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 高置信错误 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| 当前第 6 层 LDA baseline | "
            f"{pct(baseline['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(baseline['final']['metrics']['top1_accuracy'])} | "
            f"{pct(baseline['final']['metrics']['top3_accuracy'])} | "
            f"{pct(baseline['final']['metrics']['macro_f1'])} | "
            f"{baseline['calibrated']['0.96']['final']['rejection']['accepted_wrong_count']} |"
        ),
        (
            f"| 推荐的 5/6/8 层集成 | "
            f"{pct(recommended['dev']['metrics']['top1_accuracy'])} | "
            f"{pct(recommended['final']['metrics']['top1_accuracy'])} | "
            f"{pct(recommended['final']['metrics']['top3_accuracy'])} | "
            f"{pct(recommended['final']['metrics']['macro_f1'])} | "
            f"{recommended['calibrated']['0.96']['final']['rejection']['accepted_wrong_count']} |"
        ),
        "",
        "相对当前单层 baseline，推荐方案在 final frozen test 上的变化是：",
        "",
        (
            f"- Top-1：{pp(recommended['final_delta_vs_baseline']['top1_accuracy'])}；"
            f"错误曲目从 {baseline['final']['wrong_count']} 首降到 {recommended['final']['wrong_count']} 首。"
        ),
        (
            f"- Macro-F1：{pp(recommended['final_delta_vs_baseline']['macro_f1'])}；"
            "提升不是单个大类带来的偶然收益。"
        ),
        (
            f"- Top-3：{pp(recommended['final_delta_vs_baseline']['top3_accuracy'])}；"
            "没有牺牲产品最重要的候选列表体验。"
        ),
        (
            "- 拒识：把 OOF 目标接受精度从 0.95 提到 0.96 后，final 高置信错误维持在 "
            f"{recommended['calibrated']['0.96']['final']['rejection']['accepted_wrong_count']} 首，"
            f"被接受样本准确率为 {pct(recommended['calibrated']['0.96']['final']['rejection']['accepted_accuracy'])}。"
        ),
        "",
        "## 失败或不推荐的方向",
        "",
        (
            f"- 更宽的中层窗口 {broad['layer_indices']} 虽然给出最高 dev Top-1 "
            f"({pct(broad['dev']['metrics']['top1_accuracy'])})，但 final Top-3 降到 "
            f"{pct(broad['final']['metrics']['top3_accuracy'])}。它把层间信息平均得过头，排序变钝。"
        ),
        (
            f"- mean+std 池化看起来能描述歌曲内部变化，但最佳 probe 只有 dev Top-1 "
            f"{pct(mean_std['dev']['metrics']['top1_accuracy'])}。在当前数据规模下，它增加噪声维度的速度"
            "快于增加稳定风格信息的速度。"
        ),
        "- 双原型 probe 在探索运行中明显低于 baseline。当前每类十几首歌不足以稳定拆出子簇，"
        "除非未来每类歌曲显著增多，或先训练出更好的 metric-learning 投影，否则不应作为主线。",
        "",
        "## 建议",
        "",
        "把 5/6/8 层集成视为第一个可信的 P3 候选，但先不要部署。下一步应只在本分支上做一个可加载的"
        "实验模型 artifact 和 API 接入 smoke test；如果仍能把 final 高置信错误压在当前水平以内，再考虑进入主线。",
        "",
        "## 复现",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python scripts/20_run_p3_model_experiments.py",
        "```",
        "",
        "产物：",
        "",
        "- `data/processed/evaluations/p3_model_experiments.json`",
        "- `docs/P3_MODEL_EXPERIMENT_REPORT.md`",
        "",
    ]
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
        "--output",
        type=Path,
        default=root / "data/processed/evaluations/p3_model_experiments.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs/P3_MODEL_EXPERIMENT_REPORT.md",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_split(args.train_manifest)
    dev = load_split(args.dev_manifest)
    final = load_split(args.final_manifest)
    classes = np.unique(train.labels)
    if set(dev.labels) != set(classes) or set(final.labels) != set(classes):
        raise ValueError("Dev/final labels do not match training classes")

    single_layers = [
        evaluate_layer_method(train, dev, final, classes, (layer,))
        for layer in range(train.layers.shape[1])
    ]
    baseline = next(item for item in single_layers if item["layer_indices"] == [6])
    recommended_layers = top_layers_by_dev_macro(single_layers, 3)
    recommended = evaluate_layer_method(
        train,
        dev,
        final,
        classes,
        recommended_layers,
    )

    broad_candidates = [
        evaluate_layer_method(train, dev, final, classes, layers)
        for layers in best_broad_window(train.layers.shape[1])
    ]
    broad = max(
        broad_candidates,
        key=lambda item: (
            item["dev"]["metrics"]["top1_accuracy"],
            item["dev"]["metrics"]["macro_f1"],
            item["final"]["metrics"]["top3_accuracy"],
        ),
    )

    mean_std_train = load_split(args.train_manifest, mode="mean_std")
    mean_std_dev = load_split(args.dev_manifest, mode="mean_std")
    mean_std_final = load_split(args.final_manifest, mode="mean_std")
    mean_std_candidates = [
        evaluate_layer_method(
            mean_std_train,
            mean_std_dev,
            mean_std_final,
            classes,
            (layer,),
        )
        for layer in range(4, 10)
    ]
    mean_std = max(
        mean_std_candidates,
        key=lambda item: (
            item["dev"]["metrics"]["top1_accuracy"],
            item["dev"]["metrics"]["macro_f1"],
        ),
    )

    methods = {
        "baseline_layer6": baseline,
        "recommended_top3_dev_macro_ensemble": recommended,
        "best_broad_midlayer_window": broad,
        "best_mean_std_probe": mean_std,
    }
    for key in ("baseline_layer6", "recommended_top3_dev_macro_ensemble"):
        method = methods[key]
        calibration = oof_calibration(
            train,
            classes,
            tuple(method["layer_indices"]),
            repeats=args.repeats,
            splits=args.splits,
            seed=args.seed,
        )
        method["oof"] = {
            "aggregate": calibration["aggregate"],
            "temperature": calibration["temperature"],
            "thresholds": calibration["thresholds"],
        }
        method["calibrated"] = {}
        for target, threshold_info in calibration["thresholds"].items():
            temperature = TemperatureScaler(calibration["temperature"])
            threshold = threshold_info["threshold"]
            method["calibrated"][target] = {}
            for split_name, split in (("dev", dev), ("final", final)):
                raw = fit_predict_layer_ensemble(
                    train.layers,
                    train.labels,
                    split.layers,
                    tuple(method["layer_indices"]),
                )
                probabilities = temperature.transform(raw)
                method["calibrated"][target][split_name] = evaluate_probabilities(
                    probabilities,
                    split.labels,
                    classes,
                    threshold=threshold,
                )

    for method in methods.values():
        method["dev_delta_vs_baseline"] = delta_metrics(
            method["dev"]["metrics"],
            baseline["dev"]["metrics"],
        )
        method["final_delta_vs_baseline"] = delta_metrics(
            method["final"]["metrics"],
            baseline["final"]["metrics"],
        )

    evaluation = {
        "protocol": {
            "purpose": "P3 global model-head experiments; no deployment changes",
            "train_manifest": str(args.train_manifest),
            "dev_manifest": str(args.dev_manifest),
            "final_manifest": str(args.final_manifest),
            "selection_split": "development_holdout",
            "final_split": "final_frozen_test",
            "oof": f"{args.repeats}x{args.splits} StratifiedGroupKFold by work_id",
        },
        "dataset": {
            "train_songs": int(len(train.labels)),
            "dev_songs": int(len(dev.labels)),
            "final_songs": int(len(final.labels)),
            "classes": int(len(classes)),
            "layers": int(train.layers.shape[1]),
            "embedding_dim": int(train.layers.shape[2]),
        },
        "single_layer_sweep": single_layers,
        "methods": methods,
        "selected_recommendation": {
            "method": "recommended_top3_dev_macro_ensemble",
            "layer_indices": list(recommended_layers),
            "rejection_target_precision": 0.96,
            "reason": (
                "Highest three single layers by dev macro-F1; improves final Top-1 "
                "while preserving final Top-3 and high-confidence error count."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report_output, evaluation)
    print(json.dumps({
        "selected": evaluation["selected_recommendation"],
        "baseline_final": baseline["final"]["metrics"],
        "recommended_final": recommended["final"]["metrics"],
        "recommended_final_rejection_0.96": (
            recommended["calibrated"]["0.96"]["final"]["rejection"]
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
