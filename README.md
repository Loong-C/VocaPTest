# VocaP Test - 测测你的曲风最像哪位 P 主

一个 Vocaloid Producer 风格相似度系统。上传一段音乐，系统会在 41 位
P 主的参考库中返回 Top-K 风格候选，并在结果超出校准接受区域时给出低置信提示。
结果仅供娱乐，不代表模型能真正识别作曲家风格。

## 当前数据与模型

| 项目 | 当前状态 |
|---|---:|
| 训练目录 | **463 首 canonical 作品** |
| Development holdout | **78 首作品，每类 0-2 首** |
| Final frozen test | **159 首作品，每类 1-4 首** |
| 训练分段 | **5538 段** |
| Dev 分段 | **936 段** |
| Final frozen 分段 | **1890 段** |
| 音频处理 | 24kHz 单声道、20s 窗口、10s hop、均匀覆盖、最多 12 段 |
| 音频表征 | MERT-v1-95M 第 5/6/8 层，每层 768 维 |
| 分类器 | 三个歌曲均值 Shrinkage LDA head 的概率平均 |
| 置信度 | OOF probability temperature scaling + 拒识阈值 |

三个数据分区按 YouTube ID 和 VocaDB `work_id` 双重隔离：

- **Training songs**：参与当前 LDA 模型训练。
- **Dev holdout songs**：不参与训练，用于未来模型选择、错误分析和方案比较。
- **Final frozen songs**：不参与训练、模型选择或校准，只用于最终验收。

## 覆盖的 41 位 P 主

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
| なきそ | NAKISO | 14 | 164 |
| すりぃ | Surii, Three | 15 | 168 |
| R Sound Design | usugeP | 14 | 168 |
| とあ | Toa | 10 | 120 |
| てにをは | Teniwoha | 10 | 120 |
| 煮ル果実 | NILFRUITS | 10 | 120 |
| はるまきごはん | Harumaki Gohan | 9 | 108 |
| r-906 | arukuremu | 10 | 120 |
| sasakure.UK | ささくれP, sasakureP | 10 | 120 |
| Giga | ギガ | 10 | 120 |
| れるりり | rerulili, 当社比P, ToushahiP | 10 | 120 |
| みきとP | MikitoP, 愛島, Aijima | 10 | 120 |
| ひとしずくP / やま△ | ひとしずくP, HitoshizukuP, やま△, Yama△, さも, samo | 10 | 120 |
| バルーン | balloon, 須田景凪, Suda Keina | 10 | 120 |
| 黒うさP | KurousaP, くろうさP, WhiteFlame, しゃな, syana | 3 | 36 |
| mothy | 悪ノP, AkunoP, master of the heavenly yard | 10 | 120 |
| 柊マグネタイト | Hiiragi Magnetite | 10 | 120 |
| オワタP | OwataP, ガルナ, Garuna | 10 | 120 |
| ぬゆり | Nuyuri, nulut, Lanndo, go乱心P, ぬるり, Crona | 10 | 120 |

## 评估结果

| 指标 | 36 类 / 420 首 CV | 41 类 / 463 首 CV |
|---|---:|---:|
| Top-1 | 85.05% ± 0.57% | **81.56% ± 0.73%** |
| Top-3 | 92.90% ± 0.39% | **91.27% ± 0.54%** |
| Macro-F1 | 85.86% ± 0.61% | **80.34% ± 0.91%** |
| MRR | 89.60% ± 0.38% | **87.11% ± 0.42%** |

| 指标 | Dev holdout / 78 首 | Final frozen / 159 首 |
|---|---:|---:|
| Top-1 | **80.77%** | **80.50%** |
| Top-3 | **87.18%** | **93.08%** |
| Macro-F1 | **77.41%** | **78.73%** |
| MRR | **85.43%** | **86.79%** |
| 覆盖率 | **61.54%** | **62.89%** |
| 被接受样本准确率 | **95.83%** | **97.00%** |

41 类结果已经触及当前扩张停止条件：相较 36 类稳定批次，final Macro-F1 下降约 5.9 个百分点，
dev Macro-F1 低于 80%。后续部署建议使用 36 类稳定提交，41 类提交保留为 stop-point 证据。

P2 数据更难，尤其暴露出 `じん`、`すりぃ`、`Neru`、`sasakure.UK/cosMo`、
`DECO*27` 和若干跨媒体/非典型曲目的边界问题。详细错误分析见
[P2 数据扩充报告](docs/P2_DATA_EXPANSION_REPORT.md)。

## API 与前端

| 端点 | 方法 | 说明 |
|---|:---:|---|
| `/health` | GET | 健康检查和当前模型状态 |
| `/api/producers` | GET | 获取所有 P 主 |
| `/api/producers/{slug}` | GET | 获取头像、别名、训练曲、dev 曲和 final frozen 曲 |
| `/api/analyze` | POST | 同步兼容接口，上传音频并返回 Top-K 候选 |
| `/api/analyze/jobs` | POST | 创建异步分析任务，返回真实阶段进度 |
| `/api/jobs/{job_id}` | GET | 查询 received / segmenting / embedding / classifying / done 状态 |

P 主页面三段式展示：

- 学习曲目：实际参与训练。
- 开发验证曲目：用于模型开发验证。
- 最终冻结测试曲目：用于最终验收。

P 主风格标签来自 `configs/producer_style_tags.yaml` 中缓存的 VocaDB song tags，
仅用于页面展示和搜索提示，不参与模型训练或评估。维护口径见
[P 主风格标签来源](docs/STYLE_TAG_SOURCES.md)。

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
| `configs/producers.yaml` | 41 位 P 主配置 |
| `configs/producer_style_tags.yaml` | VocaDB song tags 风格标签缓存 |
| `configs/training_catalog_additions.yaml` | 人工核验的训练目录增量 |
| `configs/dev_holdout_catalog.yaml` | development holdout 配置 |
| `configs/frozen_test_catalog.yaml` | final frozen 配置 |
| `data/processed/curated/mert_95_p1/segments.jsonl` | 训练 MERT 清单 |
| `data/processed/dev_holdout/catalog.jsonl` | dev 下载清单 |
| `data/processed/dev_holdout/mert_95_layers/segments.jsonl` | dev MERT 清单 |
| `data/processed/frozen_test/catalog.jsonl` | final frozen 下载清单 |
| `data/processed/frozen_test/mert_95_layers/segments.jsonl` | final frozen MERT 清单 |
| `data/processed/evaluations/p3_layer_fusion_deploy.json` | 当前 41 类 CV 与校准结果 |
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
  --allow-variable-per-class `
  --minimum-per-class 0 `
  --exclude-catalog configs/frozen_test_catalog.yaml `
  --ffmpeg-location <ffmpeg目录>

python scripts/18_prepare_frozen_test_catalog.py `
  --catalog configs/frozen_test_catalog.yaml `
  --audio-root data/frozen_test_audio `
  --manifest-output data/processed/frozen_test/catalog.jsonl `
  --category frozen_test `
  --expected-per-class 4 `
  --allow-variable-per-class `
  --minimum-per-class 1 `
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

python scripts/21_train_p3_layer_fusion.py
python scripts/19_evaluate_frozen_test.py `
  --manifest data/processed/dev_holdout/mert_95_layers/segments.jsonl `
  --output data/processed/evaluations/p2_dev_holdout.json `
  --protocol-name p2_development_holdout `
  --expected-per-class 2 `
  --allow-variable-per-class `
  --minimum-per-class 0
python scripts/19_evaluate_frozen_test.py `
  --protocol-name p2_final_frozen_test `
  --expected-per-class 4 `
  --allow-variable-per-class `
  --minimum-per-class 1
```

## 当前机器适配

当前 RTX 4060 Ti 8GB、约 32GB 内存下，MERT-v1-95M 使用 batch size 4 稳定。
本轮新增 top100 前两批 10 位候选 P 主，其中 41 类批次已达到扩张停止条件；完整重建缓存可在本机完成。
现阶段瓶颈仍是数据口径和评估设计，不是显存。

## 已知问题

1. 41 类批次达到停止条件，尤其是 dev/final Macro-F1 低于 80%；建议部署 36 类稳定批次。
2. 黒うさP 只有 3 首训练曲和 1 首 final 曲，保留为“少量但干净”的样本，不再硬凑 dev。
3. `じん`、`Neru`、`すりぃ`、`cosMo/sasakure.UK` 仍是重点边界，需要继续靠干净曲目和拒识阈值控制。

历史 27 类结果见 [P1 报告](docs/P1_27_PRODUCERS_FROZEN_TEST_REPORT.md)。
