# P4 数据质量与全局模型头实验

本实验不新增歌曲、不删除缓存文件，也不写任何特定 P 主规则。所有候选方法只使用现有训练、dev 和 final frozen 分区。

## 结果摘要

| 训练过滤 | Dev 选择方法 | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 错误 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | lda_equal_568 | 82.98% | 79.53% | 78.42% | 88.95% | 75.55% | 41 |
| source_clean | lda_equal_top3_dev_layers | 82.98% | 79.68% | 79.47% | 89.47% | 76.31% | 39 |
| review_clean | lda_equal_568 | 82.98% | 79.68% | 78.95% | 88.95% | 76.35% | 40 |
| protocol_complete | blend_lda_centroid_top3_dev_layers | 62.77% | 54.45% | 57.37% | 69.47% | 50.43% | 81 |

## 数据过滤

`source_clean` 只排除 VocaDB 明确标为非 Original/Reprint 风险的训练歌；`review_clean` 额外排除低评分或 PV 作者需复核的训练歌；`protocol_complete` 再排除配置里缺 `vocadb_song_id` 的训练歌，仅用于衡量严格协议的样本代价。

| 训练过滤 | 保留训练歌 | 最小类样本 | 排除歌曲 |
|---|---:|---:|---:|
| raw | 574 | 9 | 0 |
| source_clean | 573 | 9 | 1 |
| review_clean | 561 | 7 | 13 |
| protocol_complete | 387 | 1 | 187 |

## 过滤后 Manifest

| 训练过滤 | Manifest | 歌曲 | 分段 | 最小类样本 |
|---|---|---:|---:|---:|
| source_clean | `F:\Personal\Code\vocaptest\data\processed\curated\p4_data_quality\source_clean\segments.jsonl` | 573 | 6857 | 9 |
| review_clean | `F:\Personal\Code\vocaptest\data\processed\curated\p4_data_quality\review_clean\segments.jsonl` | 561 | 6713 | 7 |
| protocol_complete | `F:\Personal\Code\vocaptest\data\processed\curated\p4_data_quality\protocol_complete\segments.jsonl` | 387 | 4627 | 1 |

## 候选方法

| 训练过滤 | 方法 | Dev Top-1 | Dev Top-3 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 接受准确率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | lda_equal_568 | 82.98% | 89.36% | 79.53% | 78.42% | 88.95% | 75.55% | 93.02% |
| raw | lda_equal_top3_dev_layers | 79.79% | 88.30% | 75.54% | 80.53% | 90.00% | 77.39% | 85.19% |
| raw | lda_dev_weighted_all_layers | 81.91% | 87.23% | 78.80% | 79.47% | 87.37% | 76.32% | 86.62% |
| raw | centroid_equal_top3_dev_layers | 58.51% | 77.66% | 52.87% | 58.95% | 70.00% | 55.86% | 75.00% |
| raw | blend_lda_centroid_top3_dev_layers | 79.79% | 88.30% | 75.54% | 80.53% | 90.00% | 77.39% | 85.19% |
| source_clean | lda_equal_568 | 82.98% | 89.36% | 79.53% | 78.42% | 88.42% | 75.55% | 93.02% |
| source_clean | lda_equal_top3_dev_layers | 82.98% | 89.36% | 79.68% | 79.47% | 89.47% | 76.31% | 94.40% |
| source_clean | lda_dev_weighted_all_layers | 81.91% | 87.23% | 78.80% | 79.47% | 86.84% | 76.32% | 89.36% |
| source_clean | centroid_equal_top3_dev_layers | 58.51% | 78.72% | 52.87% | 58.95% | 72.11% | 55.92% | 74.19% |
| source_clean | blend_lda_centroid_top3_dev_layers | 82.98% | 89.36% | 79.68% | 79.47% | 89.47% | 76.31% | 94.40% |
| review_clean | lda_equal_568 | 82.98% | 89.36% | 79.68% | 78.95% | 88.95% | 76.35% | 95.04% |
| review_clean | lda_equal_top3_dev_layers | 82.98% | 89.36% | 79.68% | 78.95% | 88.95% | 76.35% | 95.04% |
| review_clean | lda_dev_weighted_all_layers | 81.91% | 86.17% | 78.94% | 80.00% | 86.84% | 77.31% | 90.78% |
| review_clean | centroid_equal_top3_dev_layers | 57.45% | 77.66% | 51.77% | 57.89% | 71.58% | 55.47% | 75.76% |
| review_clean | blend_lda_centroid_top3_dev_layers | 82.98% | 89.36% | 79.68% | 78.95% | 88.95% | 76.35% | 95.04% |
| protocol_complete | lda_equal_568 | 59.57% | 70.21% | 49.10% | 59.47% | 70.00% | 52.88% | 96.88% |
| protocol_complete | lda_equal_top3_dev_layers | 61.70% | 71.28% | 52.98% | 56.84% | 66.84% | 49.59% | 95.77% |
| protocol_complete | lda_dev_weighted_all_layers | 59.57% | 68.09% | 50.12% | 60.00% | 70.00% | 52.77% | 98.11% |
| protocol_complete | centroid_equal_top3_dev_layers | 45.74% | 67.02% | 38.14% | 43.68% | 60.53% | 39.58% | 67.57% |
| protocol_complete | blend_lda_centroid_top3_dev_layers | 62.77% | 68.09% | 54.45% | 57.37% | 69.47% | 50.43% | 87.50% |
