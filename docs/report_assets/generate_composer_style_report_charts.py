"""Generate charts for the composer-style recognition audit report."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS = json.loads((OUTPUT_DIR / "audit_results.json").read_text(encoding="utf-8"))

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

BLUE = {"base": "#A3BEFA", "dark": "#2E4780"}
GOLD = {"base": "#FFE15B", "dark": "#736422"}
ORANGE = {"base": "#F0986E", "dark": "#804126"}
OLIVE = {"base": "#A3D576", "dark": "#386411"}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Segoe UI",
                "DejaVu Sans",
                "Arial",
            ],
        },
    )


def add_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=58, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=96, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.79, left=0.13, right=0.96, bottom=0.21)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.89,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def chart_evaluation_stability() -> None:
    current = RESULTS["current_split"]
    repeated = RESULTS["repeated_original_data"]["kmeans5"]
    lda_original = RESULTS["repeated_song_level_baselines"]["shrinkage_lda"]
    clean = RESULTS["strict_clean_grouped_evaluation"]
    labels = [
        "当前单次划分\nKMeans-5",
        f"原数据 {RESULTS['repeated_original_data']['splits']} 次\nKMeans-5",
        "清洗+作品分组\nKMeans-5",
        f"原数据 {RESULTS['repeated_song_level_baselines']['splits']} 次\nShrinkage LDA",
        "清洗+作品分组\nShrinkage LDA",
    ]
    means = 100 * np.array(
        [
            current["top1"],
            repeated["top1_mean"],
            clean["kmeans5"]["top1_mean"],
            lda_original["top1_mean"],
            clean["shrinkage_lda"]["top1_mean"],
        ]
    )
    lower = 100 * np.array(
        [
            current["top1_wilson_95_low"],
            repeated["top1_mean"] - repeated["top1_std"],
            clean["kmeans5"]["top1_mean"] - clean["kmeans5"]["top1_std"],
            lda_original["top1_mean"] - lda_original["top1_std"],
            clean["shrinkage_lda"]["top1_mean"]
            - clean["shrinkage_lda"]["top1_std"],
        ]
    )
    upper = 100 * np.array(
        [
            current["top1_wilson_95_high"],
            repeated["top1_mean"] + repeated["top1_std"],
            clean["kmeans5"]["top1_mean"] + clean["kmeans5"]["top1_std"],
            lda_original["top1_mean"] + lda_original["top1_std"],
            clean["shrinkage_lda"]["top1_mean"]
            + clean["shrinkage_lda"]["top1_std"],
        ]
    )
    errors = np.vstack([means - lower, upper - means])
    colors = [
        GOLD["base"],
        GOLD["base"],
        ORANGE["base"],
        BLUE["base"],
        OLIVE["base"],
    ]
    edges = [
        GOLD["dark"],
        GOLD["dark"],
        ORANGE["dark"],
        BLUE["dark"],
        OLIVE["dark"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor=edges,
        linewidth=1.0,
        width=0.68,
        yerr=errors,
        capsize=4,
        error_kw={"ecolor": TOKENS["ink"], "elinewidth": 1.0},
    )
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TOKENS["ink"],
        )
    ax.axhline(100 / 18, color=TOKENS["muted"], linestyle=":", linewidth=1.0)
    ax.text(4.48, 100 / 18 + 1.0, "随机基线 5.6%", ha="right", fontsize=8, color=TOKENS["muted"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 88)
    ax.set_ylabel("Top-1 准确率")
    ax.set_yticks(np.arange(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
    ax.grid(axis="x", visible=False)
    add_header(
        fig,
        ax,
        "评估结论随划分与数据清洗显著变化",
        "单次划分显示 Wilson 95% 区间；重复实验显示均值 ± 1 个标准差。清洗后 KMeans 回落，而 LDA 仍保持明显优势。",
    )
    fig.savefig(OUTPUT_DIR / "evaluation_stability.png", bbox_inches="tight")
    plt.close(fig)


def chart_clean_model_comparison() -> None:
    clean = RESULTS["strict_clean_grouped_evaluation"]
    models = ["KMeans-5 原型检索", "Shrinkage LDA"]
    top1 = 100 * np.array(
        [clean["kmeans5"]["top1_mean"], clean["shrinkage_lda"]["top1_mean"]]
    )
    top3 = 100 * np.array(
        [clean["kmeans5"]["top3_mean"], clean["shrinkage_lda"]["top3_mean"]]
    )
    x = np.arange(len(models))
    width = 0.31

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    bars1 = ax.bar(
        x - width / 2,
        top1,
        width,
        label="Top-1",
        color=ORANGE["base"],
        edgecolor=ORANGE["dark"],
        linewidth=1.0,
    )
    bars2 = ax.bar(
        x + width / 2,
        top3,
        width,
        label="Top-3",
        color=BLUE["base"],
        edgecolor=BLUE["dark"],
        linewidth=1.0,
    )
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TOKENS["ink"],
            )
    ax.set_xticks(x, models)
    ax.set_ylim(0, 92)
    ax.set_ylabel("歌曲级准确率")
    ax.set_yticks(np.arange(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2)
    ax.grid(axis="x", visible=False)
    add_header(
        fig,
        ax,
        "现有 embedding 上，歌曲级 LDA 是最值得优先落地的基线",
        "50 次作品组不重叠的重复划分；删除 13 个严格标题规则命中的翻唱、卡拉 OK 等样本，并合并近重复录音组。",
    )
    fig.savefig(OUTPUT_DIR / "clean_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    use_theme()
    chart_evaluation_stability()
    chart_clean_model_comparison()


if __name__ == "__main__":
    main()
