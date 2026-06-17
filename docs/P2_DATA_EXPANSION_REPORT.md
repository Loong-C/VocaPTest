# P2 数据扩充、验证分区与当前模型报告

日期：2026-06-17

## 1. 本轮目标与完成范围

本轮目标不是继续“堆一个更大的名单”，而是给后续模型更新铺一个更可靠的数据地基：

1. 新增 4 位风格差异较大的 P 主：煮ル果実、はるまきごはん、r-906、sasakure.UK。
2. 对低样本或边界困难的旧类补歌，把 Neru、じん、なきそ、すりぃ、R Sound Design 补到 16 首训练曲。
3. 建立公开可展示但不参与训练的 development holdout，每类 2 首。
4. 把 final frozen test 扩到每类 4 首，只用于最终验收。
5. 更新配置、下载校验、MERT manifest、评估结果、API schema、前端 P 主详情页、README 和测试。

最终当前系统为 **31 位 P 主、376 首训练曲、62 首 dev holdout、124 首 final frozen**。

## 2. 数据构成

| 分区 | P 主 | 歌曲 | MERT 分段 | 用途 |
|---|---:|---:|---:|---|
| Training | 31 | 376 | 4494 | 训练当前 Shrinkage LDA |
| Development holdout | 31 | 62 | 744 | 后续模型选择、错误分析、阈值方案比较 |
| Final frozen test | 31 | 124 | 1470 | 最终验收；不参与模型选择和校准 |

三个分区均按 YouTube video ID 与 VocaDB `work_id` 做双重隔离，当前重叠数为 0。
P 主详情页会同时展示三组曲目，但 UI 文案明确区分“学习曲目”“开发验证曲目”和
“最终冻结测试曲目”，避免把 dev/final 曲误理解成训练样本。

## 3. 来源核验

新增和补充曲目仍以 VocaDB 为主索引，下载前执行以下校验：

- VocaDB 歌曲条目需为 `Original`；
- 目标 P 主需在 artists 列表中具有 composer 或默认创作者关系；
- 固定 YouTube PV 需能被 yt-dlp 解析，时长满足项目约束；
- 训练、dev、final 三个分区不得共享 video ID 或 `work_id`。

本轮涉及的新增 P 主来源页：

| P 主 | VocaDB artist |
|---|---|
| 煮ル果実 | <https://vocadb.net/Ar/64434> |
| はるまきごはん | <https://vocadb.net/Ar/28208> |
| r-906 | <https://vocadb.net/Ar/66139> |
| sasakure.UK | <https://vocadb.net/Ar/51> |

例外说明：wowaka 在 YouTube Original PV 覆盖上不足，final frozen 中保留了 1 首
VocaDB `Other` PV 的同作品可访问来源。它没有进入训练集，且在 manifest 中保留
`source_reason=vocadb_other_pv`，后续可优先替换。

## 4. 工程变更

- `configs/producers.yaml` 扩展到 31 位 P 主。
- `configs/training_catalog_additions.yaml` 记录人工核验的训练增量。
- `configs/dev_holdout_catalog.yaml` 新增 development holdout。
- `configs/frozen_test_catalog.yaml` 扩展为每类 4 首 final frozen。
- `scripts/18_prepare_frozen_test_catalog.py` 泛化为可生成 dev/final 任一 heldout 分区，并支持跨 catalog 排除。
- `scripts/19_evaluate_frozen_test.py` 支持自定义 protocol name 与 expected-per-class。
- API 的 P 主详情新增 `training_songs`、`dev_songs`、`frozen_songs`；旧字段 `test_songs` 作为 final frozen 的兼容别名保留。
- 前端 P 主页分三段展示曲目，三组都可打开用于核验和学习。
- README 更新到 31 类数据和当前指标。

## 5. 当前模型与评估

默认模型没有改成复杂网络，仍保持为目前最适合本机和当前数据规模的强基线：

- MERT-v1-95M 第 6 层；
- 20 秒窗口、10 秒 hop、每首最多 12 段；
- 歌曲级 mean pooling；
- L2 normalize；
- 等类别先验 Shrinkage LDA；
- out-of-fold logits temperature scaling；
- 以高接受精度为目标的拒识阈值。

| 评估 | Top-1 | Top-3 | Macro-F1 | MRR |
|---|---:|---:|---:|---:|
| 31 类 / 376 首分组 CV | **84.95% ± 0.89%** | **93.51% ± 0.48%** | **86.17% ± 0.88%** | **89.75% ± 0.40%** |
| Dev holdout / 62 首 | **75.81%** | **80.65%** | **74.42%** | **80.32%** |
| Final frozen / 124 首 | **78.23%** | **92.74%** | **78.24%** | **86.02%** |

拒识表现：

| 评估 | 覆盖率 | 被接受样本准确率 |
|---|---:|---:|
| OOF CV | 81.28% | 95.03% |
| Dev holdout | 69.35% | 86.05% |
| Final frozen | 64.52% | 96.25% |

这组结果的含义很清楚：当前模型不是“作者鉴定器”，但作为娱乐向 Top-K 风格候选系统已经有可用性。
新增困难类和更严格 heldout 后，Top-1 从 27 类阶段下降是正常的；同时 final frozen 的
Top-3 和被接受样本准确率仍然较好，说明拒识阈值在保护用户体验。

## 6. 暴露出的主要问题

1. `じん` 仍是主要吸收类之一，`Neru -> じん/すりぃ` 是稳定边界问题。
2. `すりぃ`、`R Sound Design`、`とあ` 等流行编曲边界更密，单纯 LDA 会把部分相近时期/声音设计混在一起。
3. `sasakure.UK <-> cosMo@暴走P` 有高速电子、芯片音色和复杂节奏上的相互吸引。
4. `DECO*27`、`Ayase`、`syudou` 的跨媒体热门曲或非典型曲容易被歌曲身份、年代和制作风格影响。
5. dev holdout 每类 2 首，只能作为模型开发信号；final frozen 每类 4 首已经更稳，但仍不足以给单个 P 主下精确结论。

## 7. 当前机器适配

当前机器为 RTX 4060 Ti 8GB、约 32GB 内存。MERT-v1-95M、batch size 4、20 秒片段在本机稳定。
本轮缓存规模仍适合本地迭代：训练嵌入约 4.7 分钟，dev 嵌入约 2.6 分钟，final frozen 嵌入约
2.9 分钟，LDA 重训与 5 次重复 5 折评估约 1.5 分钟。下载耗时主要受网络、YouTube 可访问性和
ffmpeg 路径影响，不是显存瓶颈。

因此近期不建议直接转向大模型全量微调。更合适的路线是继续使用冻结 MERT 特征，先在 dev holdout 上做可解释、可快速回滚的模型实验。

## 8. 下一步建议

1. 用 dev holdout 做模型选择，不再用 final frozen 调参。
2. 针对 `じん/Neru/すりぃ` 与 `sasakure.UK/cosMo` 做分层错误分析，按时期、BPM、歌声库、曲长和编曲形态拆开看。
3. 尝试每类多子原型的歌曲级模型，例如 class-wise mixture centroid、nearest-class-mean with shrinkage、或层级分类。
4. 在不破坏当前部署复杂度的前提下，尝试 late fusion：第 6 层 LDA + 轻量 MIR 特征 + 年代/速度校正特征。
5. 继续扩数据时优先给 dev/final 补代表性作品，而不是只给训练集补热门曲；目标是每类 20-25 首训练曲、4-6 首 dev、6-8 首 final。

## 9. 参考来源

- VocaDB API：<https://vocadb.net/swagger/index.html>
- VocaDB artist pages：<https://vocadb.net/Ar/64434>、<https://vocadb.net/Ar/28208>、<https://vocadb.net/Ar/66139>、<https://vocadb.net/Ar/51>
- yt-dlp：<https://github.com/yt-dlp/yt-dlp>

