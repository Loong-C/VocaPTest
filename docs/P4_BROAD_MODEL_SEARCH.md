# P4 Broad Model Search

本轮继续遵守两条硬约束：不新增数据，不写特定 P 主规则。所有候选方法只用现有 train/dev/final 分区；dev 用于选择，final 只用于最后报告。

## 搜索范围

- 数据过滤：`raw`、`source_clean`、`review_clean`。
- 概率融合：固定层、dev Top-3、全层权重、组合搜索、几何平均、rank/Borda、temperature。
- 拼接特征分类器：Shrinkage LDA、PCA+LDA、Logistic Regression、Ridge、Linear SVM。
- Segment-level：把现有分段 embedding 当作训练样本，按歌聚合分段概率，不新增任何音频或歌曲。

## Dev 选择结果

| 过滤 | Dev 选择方法 | Family | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 错误 | Final 接受准确率 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| source_clean | lda_best_dev_combo_temperature | layer_probability_fusion | 85.11% | 81.20% | 79.47% | 87.37% | 76.49% | 39 | 94.07% |

## Final Top-1 诊断

下面这列只作为诊断，不用于正式选择；它回答“如果偷看 final，哪些方向有潜力”。

| 过滤 | Final Top-1 最高方法 | Family | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 错误 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| source_clean | lda_equal_top3_dev_layers | layer_probability_fusion | 82.98% | 79.68% | 79.47% | 89.47% | 76.31% | 39 |

## Top Candidates By Filter

| 过滤 | 方法 | Family | Extra | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 错误 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| source_clean | lda_best_dev_combo_temperature | layer_probability_fusion | `{"layers": [1, 5, 6, 7], "temperature": 2.0, "weights": [0.25, 0.25, 0.25, 0.25]}` | 85.11% | 81.20% | 79.47% | 87.37% | 76.49% | 39 |
| source_clean | concat_pca_lda | concat_classifier | `{"feature_set": "single_6", "layers": [6], "pca_dim": null}` | 84.04% | 80.20% | 77.37% | 87.89% | 74.32% | 43 |
| source_clean | concat_ridge | concat_classifier | `{"alpha": 100.0, "feature_set": "single_6", "layers": [6], "pca_dim": null}` | 85.11% | 80.00% | 78.42% | 90.00% | 74.53% | 41 |
| source_clean | lda_best_dev_combo_arithmetic | layer_probability_fusion | `{"fusion": "arithmetic", "layers": [3, 6, 10], "max_combo_size": 3, "rank_scale": null, "weights": [0.333333333333333...` | 84.04% | 79.87% | 76.32% | 88.42% | 74.29% | 45 |
| source_clean | lda_best_dev_combo_geometric | layer_probability_fusion | `{"fusion": "geometric", "layers": [2, 6, 7], "max_combo_size": 3, "rank_scale": null, "weights": [0.3333333333333333,...` | 82.98% | 79.73% | 77.89% | 89.47% | 74.80% | 42 |
| source_clean | lda_equal_top3_dev_layers | layer_probability_fusion | `{"fusion": "arithmetic", "layers": [5, 6, 7], "weights": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]}` | 82.98% | 79.68% | 79.47% | 89.47% | 76.31% | 39 |
| source_clean | lda_equal_568 | layer_probability_fusion | `{"fusion": "arithmetic", "layers": [5, 6, 8], "weights": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]}` | 82.98% | 79.53% | 78.42% | 88.42% | 75.55% | 41 |
| source_clean | concat_linear_svc | concat_classifier | `{"C": 0.01, "feature_set": "single_6", "layers": [6], "pca_dim": null}` | 82.98% | 79.27% | 78.42% | 87.89% | 74.21% | 41 |
| source_clean | lda_dev_weighted_all_layers | layer_probability_fusion | `{"fusion": "arithmetic", "layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "weights": [0.08985724379392969, 0.0, ...` | 81.91% | 78.80% | 79.47% | 86.84% | 76.40% | 39 |
| source_clean | lda_best_dev_combo_rank | layer_probability_fusion | `{"fusion": "rank", "layers": [5, 6], "max_combo_size": 3, "rank_scale": 8.0, "weights": [0.5, 0.5]}` | 81.91% | 78.13% | 76.32% | 88.42% | 73.38% | 45 |
| source_clean | lda_equal_all_layers | layer_probability_fusion | `{"fusion": "arithmetic", "layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "weights": [0.07692307692307693, 0.076...` | 80.85% | 77.28% | 78.95% | 86.84% | 77.39% | 40 |
| source_clean | lda_dev_weighted_geomean_all_layers | layer_probability_fusion | `{"fusion": "geometric", "layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "weights": [0.16131142523080394, 0.0253...` | 78.72% | 74.93% | 78.42% | 88.42% | 75.47% | 41 |
| source_clean | concat_logreg | concat_classifier | `{"C": 0.3, "feature_set": "default_568", "layers": [5, 6, 8], "pca_dim": null}` | 78.72% | 72.88% | 75.26% | 85.79% | 70.42% | 47 |

## 数据过滤摘要

| 过滤 | 训练歌曲 | 最小类样本 | 排除歌曲 | 排除原因 |
|---|---:|---:|---:|---|
| source_clean | 573 | 9 | 1 | `{"configured_source_not_original": 1}` |

## 复现

```powershell
$env:PYTHONPATH='src'
python scripts/27_run_p4_broad_model_search.py
```

