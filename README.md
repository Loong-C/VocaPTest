# VocaP Test - 测测你的曲风最像哪位 P 主

一个 Vocaloid Producer 风格相似度系统。上传一段音乐，系统会在 31 位
P 主的参考库中返回 Top-K 风格候选，并在结果超出校准接受区域时给出低置信提示。
结果仅供娱乐，不代表模型能真正识别作曲家风格。

## 当前数据与模型

| 项目 | 当前状态 |
|---|---:|
| 训练目录 | **376 首 canonical 作品** |
| Development holdout | **62 首作品，每类 2 首** |
| Final frozen test | **124 首作品，每类 4 首** |
| 训练分段 | **4494 段** |
| Dev 分段 | **744 段** |
| Final frozen 分段 | **1470 段** |
| 音频处理 | 24kHz 单声道、20s 窗口、10s hop、均匀覆盖、最多 12 段 |
| 音频表征 | MERT-v1-95M 第 6 层，768 维 |
| 分类器 | 歌曲均值 + 等先验 Shrinkage LDA |
| 置信度 | OOF logits temperature scaling + 拒识阈值 |

三个数据分区按 YouTube ID 和 VocaDB `work_id` 双重隔离：

- **Training songs**：参与当前 LDA 模型训练。
- **Dev holdout songs**：不参与训练，用于未来模型选择、错误分析和方案比较。
- **Final frozen songs**：不参与训练、模型选择或校准，只用于最终验收。

## 覆盖的 31 位 P 主

| P 主 | 别名 | 训练歌曲 | 训练分段 |
|---|---|---:|---:|
| wowaka | 現実逃避P, GenjitsutouhiP | 10 | 120 |
| kemu | 堀江晶太 | 12 | 144 |
| Neru | 押入れP | 16 | 192 |
| DECO*27 | - | 14 | 168 |
| ピノキオピー | PinocchioP, ピノキオP | 13 | 155 |
| Mitchie M | - | 10 | 120 |
| じん | 自然の敵P, Jin | 16 | 192 |
| Orangestar | 蜜柑星P | 12 | 144 |
| cosMo@暴走P | cosMo, 暴走P | 13 | 156 |
| ハチ | Hachi, 米津玄師 | 12 | 144 |
| 40mP | 40㍍P | 12 | 144 |
| ナユタン星人 | NayutalieN | 11 | 132 |
| かいりきベア | Kairiki Bear | 12 | 144 |
| Kanaria | - | 10 | 119 |
| Chinozo | - | 13 | 156 |
| 稲葉曇 | Inabakumori | 14 | 168 |
| MIMI | mimi_3mi | 14 | 168 |
| MARETU | 極悪P | 12 | 144 |
| n-buna | ナブナ, Nabuna | 11 | 132 |
| Ayase | Ayase_0404 | 11 | 132 |
| いよわ | iyowa | 10 | 120 |
| syudou | しゅどう | 10 | 120 |
| なきそ | NAKISO | 16 | 188 |
| すりぃ | Surii, Three | 16 | 180 |
| R Sound Design | usugeP | 16 | 192 |
| とあ | Toa | 10 | 120 |
| てにをは | Teniwoha | 10 | 120 |
| 煮ル果実 | NILFRUITS | 10 | 120 |
| はるまきごはん | Harumaki Gohan | 10 | 120 |
| r-906 | arukuremu | 10 | 120 |
| sasakure.UK | ささくれP, sasakureP | 10 | 120 |

## 评估结果

| 指标 | 27 类 / 309 首 CV | 31 类 / 376 首 CV |
|---|---:|---:|
| Top-1 | 86.73% ± 0.86% | **84.95% ± 0.89%** |
| Top-3 | 93.92% ± 0.58% | **93.51% ± 0.48%** |
| Macro-F1 | 87.00% ± 0.80% | **86.17% ± 0.88%** |
| MRR | 90.98% ± 0.44% | **89.75% ± 0.40%** |

| 指标 | Dev holdout / 62 首 | Final frozen / 124 首 |
|---|---:|---:|
| Top-1 | **75.81%** | **78.23%** |
| Top-3 | **80.65%** | **92.74%** |
| Macro-F1 | **74.42%** | **78.24%** |
| MRR | **80.32%** | **86.02%** |
| 覆盖率 | **69.35%** | **64.52%** |
| 被接受样本准确率 | **86.05%** | **96.25%** |

P2 数据更难，尤其暴露出 `じん`、`すりぃ`、`Neru`、`sasakure.UK/cosMo`、
`DECO*27` 和若干跨媒体/非典型曲目的边界问题。详细错误分析见
[P2 数据扩充报告](docs/P2_DATA_EXPANSION_REPORT.md)。

## API 与前端

| 端点 | 方法 | 说明 |
|---|:---:|---|
| `/health` | GET | 健康检查和当前模型状态 |
| `/api/producers` | GET | 获取所有 P 主 |
| `/api/producers/{slug}` | GET | 获取头像、别名、训练曲、dev 曲和 final frozen 曲 |
| `/api/analyze` | POST | 上传音频并返回 Top-K 候选 |

P 主页面三段式展示：

- 学习曲目：实际参与训练。
- 开发验证曲目：用于模型开发验证。
- 最终冻结测试曲目：用于最终验收。

前端导航保留首页、分析页和 P 主图鉴页；顶部 GitHub 图标指向仓库，不再提供独立“关于”页面。

## VPS 部署

完整部署会拉取指定分支、安装依赖、构建前端、安装 systemd 服务并刷新 Nginx：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_vps.ps1
```

页面或文档更新可使用较快的更新路径，跳过系统包、Python 依赖和服务安装，只重新拉取代码并构建前端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_vps.ps1 `
  -SkipModelSync -SkipSystemPackages -SkipPythonDeps -SkipServiceInstall -SkipNginxInstall
```

脚本默认在 VPS 上以后台任务执行更新，并轮询 `/tmp/vocaptest-deploy-*.log`；如需保持单个 SSH 前台会话，可追加 `-RunUpdateInForeground`。

生产部署包含：
- `/VocaPTest/api/analyze` 每 IP 每分钟 10 次、突发 5 次的 Nginx 限流。
- HSTS、CSP、X-Frame-Options、X-Content-Type-Options 等基础安全响应头。
- FastAPI 服务以 `vocaptest` 低权限用户运行。

VPS 首次安全加固可执行：

```bash
bash deploy/harden_vps_security.sh
```

该脚本会禁用 SSH 密码登录、保留 root 公钥登录、开启 UFW，仅放行 22/80/443，并启用 fail2ban。

## 关键文件

| 文件 | 说明 |
|---|---|
| `configs/producers.yaml` | 31 位 P 主配置 |
| `configs/training_catalog_additions.yaml` | 人工核验的训练目录增量 |
| `configs/dev_holdout_catalog.yaml` | development holdout 配置 |
| `configs/frozen_test_catalog.yaml` | final frozen 配置 |
| `data/processed/curated/mert_95_p1/segments.jsonl` | 训练 MERT 清单 |
| `data/processed/dev_holdout/catalog.jsonl` | dev 下载清单 |
| `data/processed/dev_holdout/mert_95_layers/segments.jsonl` | dev MERT 清单 |
| `data/processed/frozen_test/catalog.jsonl` | final frozen 下载清单 |
| `data/processed/frozen_test/mert_95_layers/segments.jsonl` | final frozen MERT 清单 |
| `data/processed/evaluations/p1_selected_layer.json` | 当前 31 类 CV 与校准结果 |
| `data/processed/evaluations/p2_dev_holdout.json` | dev holdout 结果 |
| `data/processed/evaluations/p1_frozen_test.json` | 当前 final frozen 结果 |

## 复现数据与评估

```powershell
python scripts/16_expand_training_catalog.py --ffmpeg-location <ffmpeg目录>

python scripts/18_prepare_frozen_test_catalog.py `
  --catalog configs/dev_holdout_catalog.yaml `
  --audio-root data/dev_holdout_audio `
  --manifest-output data/processed/dev_holdout/catalog.jsonl `
  --category dev_holdout `
  --expected-per-class 2 `
  --exclude-catalog configs/frozen_test_catalog.yaml `
  --ffmpeg-location <ffmpeg目录>

python scripts/18_prepare_frozen_test_catalog.py `
  --catalog configs/frozen_test_catalog.yaml `
  --audio-root data/frozen_test_audio `
  --manifest-output data/processed/frozen_test/catalog.jsonl `
  --category frozen_test `
  --expected-per-class 4 `
  --exclude-catalog configs/dev_holdout_catalog.yaml `
  --ffmpeg-location <ffmpeg目录>

python scripts/09_rebuild_p1_layer_embeddings.py

python scripts/09_rebuild_p1_layer_embeddings.py `
  --decisions data/processed/dev_holdout/catalog.jsonl `
  --audio-root data/dev_holdout_audio `
  --embedding-output data/processed/embeddings/mert_95_dev_holdout_layers `
  --manifest-output data/processed/dev_holdout/mert_95_layers/segments.jsonl

python scripts/09_rebuild_p1_layer_embeddings.py `
  --decisions data/processed/frozen_test/catalog.jsonl `
  --audio-root data/frozen_test_audio `
  --embedding-output data/processed/embeddings/mert_95_frozen_layers `
  --manifest-output data/processed/frozen_test/mert_95_layers/segments.jsonl

python scripts/13_train_p1_selected_layer.py
python scripts/19_evaluate_frozen_test.py `
  --manifest data/processed/dev_holdout/mert_95_layers/segments.jsonl `
  --output data/processed/evaluations/p2_dev_holdout.json `
  --protocol-name p2_development_holdout `
  --expected-per-class 2
python scripts/19_evaluate_frozen_test.py `
  --protocol-name p2_final_frozen_test `
  --expected-per-class 4
```

## 当前机器适配

当前 RTX 4060 Ti 8GB、约 32GB 内存下，MERT-v1-95M 使用 batch size 4 稳定。
本轮新增 67 首训练曲、62 首 dev 和 70 首新增 final frozen 的缓存均可在本机完成。
现阶段瓶颈仍是数据口径和评估设计，不是显存。

## 已知问题

1. Dev holdout 每类只有 2 首，适合模型开发信号，不适合单类精确结论。
2. Final frozen 每类 4 首仍偏小，但已经比 2 首阶段稳定。
3. wowaka 作品 YouTube Original PV 覆盖不足，final frozen 中包含 1 首 VocaDB `Other` PV，报告中单独标注。
4. `じん` 仍是主要吸收类；`Neru -> じん/すりぃ`、`cosMo <-> sasakure.UK` 是重点边界。
5. 当前拒识阈值在 final frozen 上较保守，覆盖 64.52%，但被接受样本准确率为 96.25%。

历史 27 类结果见 [P1 报告](docs/P1_27_PRODUCERS_FROZEN_TEST_REPORT.md)。
