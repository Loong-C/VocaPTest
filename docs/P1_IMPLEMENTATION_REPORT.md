# P1 实施报告：MERT 中层选择、校准与特征消融

日期：2026-06-11

## 最终结论

P1 默认模型采用：

- MERT-v1-95M 第 6 个 hidden state；
- 24 kHz 单声道、20 秒窗口、10 秒步长；
- 通过 RMS 门槛后均匀覆盖整首歌，最多 12 段；
- 每首歌对片段 embedding 求均值并 L2 归一化；
- 等类别先验 Shrinkage LDA；
- 基于重复 OOF `decision_function` logits 的 temperature scaling；
- 以 95% OOF 接受精度为目标的拒识阈值。

没有采用多层融合、多统计池化、MIR 或投影头，因为它们均未在相同分组
交叉验证中超过第 6 层歌曲均值 LDA。

## 数据与评估协议

- 174 首清洗后的 canonical 歌曲；
- 18 位作曲家；
- 2086 个均匀时间采样片段；
- 13 层 MERT 池化 embedding；
- 按 `work_id` 分组的 `StratifiedGroupKFold`；
- 最终模型报告使用 5 次重复 5 折。

## 最终结果

| 指标 | P0 最后一层 LDA | P1 第 6 层 LDA |
|---|---:|---:|
| Top-1 | 62.47% ± 1.99% | **87.47% ± 1.25%** |
| Top-3 | 79.48% ± 2.25% | **95.63% ± 1.38%** |
| Macro-F1 | 59.73% ± 2.40% | **85.70% ± 1.67%** |
| MRR | 72.94% ± 1.49% | **91.72% ± 0.70%** |

Top-1 相对 P0 提升 25.00 个百分点。

## 层与特征消融

| 方案 | Top-1 |
|---|---:|
| 第 6 层歌曲均值 + Shrinkage LDA | **87.93%** |
| 嵌套非负多层概率融合 | 86.78% |
| 128 维正则化投影 + supervised contrastive | 83.52% |
| 第 6 层均值 + PCA-64 | 81.61% |
| 第 6 层均值 + MIR + PCA-64 | 81.61% |
| 均值 + 标准差 + PCA-64 | 79.50% |
| 多统计池化 + PCA-64 | 78.16% |
| 均值 + 标准差 + 段间变化 + PCA-64 | 77.20% |
| 44 维 MIR 单独 | 41.57% |

多统计与 MIR 的负结果被保留，避免后续重复引入无收益的复杂度。

## 校准与拒识

- 温度：`18.4117`；
- 未校准 OOF log-loss：`3.0898`；
- 校准后 OOF log-loss：`0.5128`；
- 接受阈值：`0.70765`；
- OOF 覆盖率：`77.59%`；
- OOF 接受样本精度：`95.11%`。

API 现在返回：

- `accepted`
- `confidence`
- `margin`
- `entropy`

真实端到端检查中，保留的 40mP 原曲被正确接受；数据清洗阶段排除的错误
实体样本被判为低置信并拒识。该阈值仍不是严格的开放集保证，后续应使用
真正的库外冻结测试集验证。

## 修复的工程问题

1. 通用 split 改为确定性、逐作者分层，输入顺序不再影响固定 seed。
2. MLP probe 验证与早停改为歌曲级 macro-F1。
3. 余弦相似度文档与测试修正为 `[-1, 1]`，不再称为概率。
4. 片段超过上限时改为均匀覆盖整首歌，而不是只取最响片段。
5. 一次 MERT 前向缓存全部 13 层，批量大小 4 适配 8GB 显存。
6. 校准从饱和的 `predict_proba` 改为原始 LDA logits。
7. API 使用校准概率，并对低置信结果给出显式拒识警告。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\09_rebuild_p1_layer_embeddings.py
.\.venv\Scripts\python.exe scripts\10_train_layer_fusion.py
.\.venv\Scripts\python.exe scripts\11_extract_mir_features.py
.\.venv\Scripts\python.exe scripts\12_evaluate_p1_feature_ablations.py
.\.venv\Scripts\python.exe scripts\13_train_p1_selected_layer.py
.\.venv\Scripts\python.exe scripts\14_evaluate_projection_head.py
.\.venv\Scripts\python.exe -m pytest -q
```

模型文件生成于：

`data/processed/models/p1_selected_layer_lda.pkl`

该 pickle 按仓库规则不提交；缺少模型时 `/health` 返回 `degraded`。

## 局限

- 第 6 层是在当前 174 首数据上选择的，尚无永久冻结外部测试集；
- wowaka 与 Neru 各只有 5 首，类别置信区间仍偏宽；
- 数据来源频道、歌声库、母带风格仍可能成为作曲家标签的代理变量；
- P1 不应被解释为已解决开放集识别或真正的“作曲技法理解”。
