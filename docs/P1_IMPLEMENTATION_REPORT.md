# P1 实施报告：MERT 中层选择、校准与特征消融

日期：2026-06-13

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

- 239 首清洗后的 canonical 歌曲；
- 20 位作曲家；
- 2866 个均匀时间采样片段；
- 13 层 MERT 池化 embedding；
- 按 `work_id` 分组的 `StratifiedGroupKFold`；
- 最终模型报告使用 5 次重复 5 折。

## 最终结果

| 指标 | 18 类 / 174 首 | 20 类 / 192 首 | 当前 20 类 / 239 首 |
|---|---:|---:|---:|
| Top-1 | 87.47% ± 1.25% | 86.67% ± 1.36% | **88.20% ± 0.55%** |
| Top-3 | 95.63% ± 1.38% | **96.35% ± 0.74%** | 95.56% ± 0.56% |
| Macro-F1 | 85.70% ± 1.67% | 85.64% ± 1.52% | **88.48% ± 0.55%** |
| MRR | 91.72% ± 0.70% | 91.62% ± 0.76% | **92.30% ± 0.29%** |

本轮不是平均堆数据：所有类别至少增加 2 首，wowaka 和 Neru 从 5 首补到
10 首，Kanaria 从 7 首补到 10 首。wowaka OOF F1 从约 0.73 升至 0.89，
Neru 从约 0.62 升至 0.75，说明低样本类补齐确实改善了泛化。

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

在 239 首数据上重新运行 3 次重复 5 折的全层嵌套评估，第 6 层 Top-1
为 88.28%、Top-3 为 95.82%；非负多层融合 Top-1 为 87.87%、Top-3
为 93.58%。融合更慢且更差，因此继续部署第 6 层。

后续又使用相同的 3 次重复 5 折外层协议，评估 RDA、每类双原型和可学习
注意力池化。三者 Top-1 分别为 82.71%、86.75% 和 87.17%，均未超过
基线的 88.28%。详细协议和失败原因见
[P1 模型变体实验](P1_MODEL_VARIANT_EXPERIMENTS.md)。

## 校准与拒识

- 温度：`13.6660`；
- 未校准 OOF log-loss：`2.8235`；
- 校准后 OOF log-loss：`0.4516`；
- 接受阈值：`0.68299`；
- OOF 覆盖率：`83.51%`；
- OOF 接受样本精度：`95.09%`。

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
7. API 使用校准概率，并对低置信结果给出显式拒识状态。
8. 前端不再直接展示黄色英文警告框，而将拒识解释为中文“参考结果”。
9. P 主详情接口从当前训练 manifest 生成曲目列表，避免静态展示与模型脱节。
10. 20 张头像本地化存储，来源页固定为 VocaDB，避免运行时热链失败。

## 目录扩充

累计新增 65 首人工核验作品：n-buna 与 Ayase 各 11 首，其余 18 位 P 主
至少 2 首；最终每类 10–14 首。每首新增歌曲均固定：

- VocaDB `Original` song ID；
- YouTube video ID；
- 允许的频道 ID 和来源等级；
- 独立 `work_id`。

下载脚本会在入库前校验频道与时长，不再把普通搜索结果自动标记为 accepted。
64 首使用官方上传；kemu 的早期作品《ぼくらの報復政策》已无官方 YouTube
上传，使用 VocaDB 明确列为 Reprint 的镜像，并在 manifest 中标记
`source_kind=vocadb_reprint`。同时修正了 MIMI 同名艺术家的 VocaDB ID 和头像。
详细清单和本次工程汇总见
[目录与界面更新报告](P1_CATALOG_UI_UPDATE_REPORT.md)。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\16_expand_training_catalog.py `
  --ffmpeg-location <包含 ffmpeg 和 ffprobe 的目录>
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

- 第 6 层虽在 20 类、239 首数据上复核，仍无永久冻结外部测试集；
- じん当前 OOF F1 为 58.75%，继续加歌前应先审计是否混入跨度过大的项目曲、
  角色曲或合作作品；
- Neru recall 为 68%，其中 28% 的 OOF 样本被判为 じん；
- 数据来源频道、歌声库、母带风格仍可能成为作曲家标签的代理变量；
- P1 不应被解释为已解决开放集识别或真正的“作曲技法理解”。
