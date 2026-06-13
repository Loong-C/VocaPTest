# VocaP Test - 测测你的曲风最像哪位 P 主

一个娱乐向的 Vocaloid Producer 风格相似度系统。上传一段音乐，系统会在 27 位
P 主的参考库中返回 Top-K 风格候选，并在结果超出校准接受区域时给出低置信提示。

## 当前数据与模型

| 项目 | 当前状态 |
|---|---:|
| 训练目录 | **309 首 canonical 作品** |
| 冻结测试目录 | **54 首作品，每类 2 首** |
| 训练分段 | **3690 段** |
| 冻结测试分段 | **642 段** |
| 音频处理 | 24kHz 单声道、20s 窗口、10s hop、均匀覆盖、最多 12 段 |
| 音频表征 | MERT-v1-95M 第 6 层，768 维 |
| 分类器 | 歌曲均值 + 等先验 Shrinkage LDA |
| 置信度 | OOF logits temperature scaling + 拒识阈值 |

冻结测试曲与训练曲按 YouTube ID 和 VocaDB `work_id` 双重隔离，从不参与训练、
层选择或阈值校准。曲目均通过 VocaDB `Original` 条目、作曲者角色和 YouTube
原始 PV 关系核验。

## 覆盖的 27 位 P 主

| P 主 | 别名 | 训练歌曲 | 训练分段 |
|---|---|---:|---:|
| wowaka | 現実逃避P | 10 | 120 |
| kemu | - | 12 | 144 |
| Neru | 押入れP | 10 | 120 |
| DECO*27 | - | 14 | 168 |
| ピノキオピー | - | 13 | 155 |
| Mitchie M | - | 10 | 120 |
| じん | - | 13 | 156 |
| Orangestar | - | 12 | 144 |
| cosMo@暴走P | - | 13 | 156 |
| ハチ | - | 12 | 144 |
| 40mP | - | 12 | 144 |
| ナユタン星人 | - | 11 | 132 |
| かいりきベア | - | 12 | 144 |
| Kanaria | - | 10 | 119 |
| Chinozo | - | 13 | 156 |
| 稲葉曇 | - | 14 | 168 |
| MIMI | - | 14 | 168 |
| MARETU | - | 12 | 144 |
| n-buna | ナブナ | 11 | 132 |
| Ayase | - | 11 | 132 |
| いよわ | - | 10 | 120 |
| syudou | - | 10 | 120 |
| なきそ | - | 10 | 116 |
| すりぃ | - | 10 | 108 |
| R Sound Design | - | 10 | 120 |
| とあ | - | 10 | 120 |
| てにをは | - | 10 | 120 |

每位 P 主另有 2 首冻结测试曲，可在 P 主详情页中查看；冻结曲不会被列入训练统计。

## 评估结果

分组交叉验证使用 5 次重复 5 折 `StratifiedGroupKFold`，按 `work_id` 隔离：

| 指标 | 20 类 / 239 首 | 27 类 / 309 首 |
|---|---:|---:|
| Top-1 | 88.20% ± 0.55% | **86.73% ± 0.86%** |
| Top-3 | 95.56% ± 0.56% | **93.92% ± 0.58%** |
| Macro-F1 | 88.48% ± 0.55% | **87.00% ± 0.80%** |
| MRR | 92.30% ± 0.29% | **90.98% ± 0.44%** |

扩展到 27 类后，任务明显更难，但指标只小幅下降。OOF 拒识覆盖率为 81.04%，
被接受样本精度为 95.05%。

首次冻结测试只评估一次，不据此回调模型：

| 指标 | 54 首冻结测试 |
|---|---:|
| Top-1 | **87.04%** |
| Top-3 | **92.59%** |
| Macro-F1 | **84.59%** |
| MRR | **89.73%** |
| 拒识覆盖率 | **72.22%** |
| 被接受样本准确率 | **97.44%** |

冻结集暴露了有价值的弱点：なきそ和すりぃ各 2 首均未命中，じん吸收了多首其他
摇滚/短篇编曲作品；唯一被高置信接受的错误是すりぃ被判为とあ。这些失败被原样
保留，避免把冻结集变成调参集。

详细结果见 [27 类与冻结测试总报告](docs/P1_27_PRODUCERS_FROZEN_TEST_REPORT.md)。

## API 与前端

| 端点 | 方法 | 说明 |
|---|:---:|---|
| `/health` | GET | 健康检查和当前模型状态 |
| `/api/producers` | GET | 获取所有 P 主 |
| `/api/producers/{slug}` | GET | 获取头像、别名、训练曲与冻结测试曲 |
| `/api/analyze` | POST | 上传音频并返回 Top-K 候选 |

P 主页面使用本地化头像，并明确区分可点击的训练曲和冻结测试曲。

```powershell
# 后端
python scripts/06_run_api.py

# 前端
Set-Location web
npm run dev
```

## 关键文件

| 文件 | 说明 |
|---|---|
| `configs/producers.yaml` | 27 位 P 主、别名和来源配置 |
| `configs/training_catalog_additions.yaml` | 人工核验的训练目录增量 |
| `configs/frozen_test_catalog.yaml` | 严格隔离的冻结测试目录 |
| `data/processed/curated/mert_95_p1/segments.jsonl` | 训练 MERT 缓存清单 |
| `data/processed/frozen_test/catalog.jsonl` | 冻结测试下载清单 |
| `data/processed/frozen_test/mert_95_layers/segments.jsonl` | 冻结测试 MERT 清单 |
| `data/processed/evaluations/p1_selected_layer.json` | 27 类分组 CV 与校准结果 |
| `data/processed/evaluations/p1_frozen_test.json` | 冻结测试结果 |
| `data/processed/models/p1_selected_layer_lda.pkl` | 本地生成的默认模型，不提交 Git |

## 复现数据与评估

```powershell
# 核验并补充训练目录
python scripts/16_expand_training_catalog.py --ffmpeg-location <ffmpeg目录>

# 核验并准备冻结测试目录
python scripts/18_prepare_frozen_test_catalog.py --ffmpeg-location <ffmpeg目录>

# 训练集全层 MERT 缓存
python scripts/09_rebuild_p1_layer_embeddings.py

# 冻结集全层 MERT 缓存
python scripts/09_rebuild_p1_layer_embeddings.py `
  --decisions data/processed/frozen_test/catalog.jsonl `
  --audio-root data/frozen_test_audio `
  --embedding-output data/processed/embeddings/mert_95_frozen_layers `
  --manifest-output data/processed/frozen_test/mert_95_layers/segments.jsonl

# 训练、交叉验证和冻结评估
python scripts/13_train_p1_selected_layer.py
python scripts/19_evaluate_frozen_test.py
```

## 当前机器适配

当前 RTX 4060 Ti 8GB、约 32GB 内存下，MERT-v1-95M 使用 batch size 4 稳定运行。
本轮 70 首新增训练曲的缓存约 4 分钟，54 首冻结曲约 4 分钟，LDA 训练与重复评估
约 1 分钟。现阶段瓶颈不是显存，而是每类作品覆盖、时期偏移和独立评估样本数。

## 已知问题

1. 冻结集目前仅每类 2 首，逐类结果方差很大，下一步应扩至每类 4-5 首。
2. 拒识由库内 OOF 校准，不构成完整的开放集识别保证。
3. 歌声库、上传频道、母带与作品时期仍可能成为作者标签的代理变量。
4. じん的 OOF F1 仅 54.66%，类别内部跨度和与其他摇滚作者的边界仍需审计。
5. 冻结集中なきそ、すりぃ均为 0/2，提示短篇编曲和时期变化尚未被训练集覆盖。
6. Neru 的 OOF 样本约 26% 被判为じん，是当前最明显的成对混淆之一。

项目结构与早期设计见 [agent.md](agent.md)，历史 P1 消融见
[P1 实施报告](docs/P1_IMPLEMENTATION_REPORT.md)。
