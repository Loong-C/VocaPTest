# VocaP Test

测试你的曲风最像哪位 Vocaloid P 主。

这是一个娱乐向的 Vocaloid Producer 风格相似度系统。用户上传一段音乐后，系统会用冻结的 MERT-v1-95M 音频表征，在 50 位 P 主的参考库中返回 Top-K 风格候选，并在置信度不足时给出低置信提示。结果只表示“风格相似”，不代表模型真的识别作曲者身份。

## 当前模型

线上默认后端是 **P4 calibrated stacking**：

| 项目 | 当前状态 |
|---|---:|
| 训练集 | 573 首 source-clean 训练作品 |
| Dev holdout | 94 首作品 |
| Final frozen test | 190 首作品 |
| 覆盖 P 主 | 50 位 |
| 音频切分 | 24kHz mono, 20s window, 10s hop, max 12 segments |
| 音频表征 | MERT-v1-95M 全层缓存 |
| 部署 artifact | `data/processed/models/p4_calibrated_stacking.pkl` |
| API 后端标识 | `mert_95_p4_calibrated_stacking` |

模型使用多个全局 LDA 基头做 stacking：

- 单层头：layer 6, 7, 8
- 层融合头：5/6/7, 5/6/8
- concat 头：5/6, 6/7, 5/6/7, 5/6/8, 6/7/8, 7/8/9, 4/5/6/7/8
- meta model：`LogisticRegression(C=0.03)` on log-probability features
- 选择协议：train-only grouped CV 以 log-loss/MRR 为主，dev 只做守门，final 只报告

## 评估结果

Raw baseline final：

| Metric | Raw |
|---|---:|
| Top-1 | 78.42% |
| Top-3 | 88.95% |
| Macro-F1 | 75.55% |

P4 calibrated stacking final：

| Metric | P4 | Delta |
|---|---:|---:|
| Top-1 | 82.11% | +3.68 pp |
| Top-3 | 90.00% | +1.05 pp |
| Macro-F1 | 80.00% | +4.45 pp |
| MRR | 86.44% | n/a |
| Log loss | 0.9116 | n/a |

这版没有用 final 调参。它满足 `macro_f1_plus_4pp_guarded` 门槛：Macro-F1 提升超过 4pp，同时 Top-1 和 Top-3 均不退化。

详细报告：

- [P4 总览](docs/P4_MODEL_SEARCH_SUMMARY.md)
- [P4 calibrated stacking](docs/P4_CALIBRATED_STACKING.md)
- [P4 deploy 评估 JSON](data/processed/evaluations/p4_calibrated_stacking_deploy.json)

## API

| Endpoint | Method | 说明 |
|---|:---:|---|
| `/health` | GET | 健康检查和当前后端 |
| `/api/producers` | GET | 获取全部 P 主 |
| `/api/producers/{slug}` | GET | 获取 P 主头像、别名、训练/dev/final 曲目 |
| `/api/analyze` | POST | 同步上传音频并返回 Top-K 候选 |
| `/api/analyze/jobs` | POST | 创建异步分析任务 |
| `/api/jobs/{job_id}` | GET | 查询 received / segmenting / embedding / classifying / done 状态 |

生产环境路径为：

```text
https://linkukai.com/VocaPTest/
```

## 本地运行

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

训练当前部署模型：

```powershell
python scripts/35_train_p4_calibrated_stacking.py
```

启动 API：

```powershell
python scripts/06_run_api.py
```

前端：

```powershell
cd web
npm install
npm run dev
```

## VPS 部署

完整部署会拉取指定分支、安装依赖、构建前端、安装 systemd 服务、刷新 Nginx，并同步本地 `data/processed/models/*.pkl` 模型 artifact：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_vps.ps1
```

只更新代码、前端和模型时使用快速路径：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_vps.ps1 `
  -SkipSystemPackages -SkipPythonDeps -SkipServiceInstall -SkipNginxInstall
```

如果只改页面或文档、不需要同步模型，可额外加：

```powershell
-SkipModelSync
```

部署脚本默认部署 `master` 分支，默认服务器为 `root@187.77.136.20`，默认应用目录为 `/srv/vocaptest/app`。部署完成后会检查 systemd 状态和：

```text
http://127.0.0.1:8000/health
```

VPS 首次安全加固：

```bash
bash deploy/harden_vps_security.sh
```

## 关键文件

| 文件 | 说明 |
|---|---|
| `configs/retrieval.yaml` | 当前默认检索后端和模型路径 |
| `configs/producers.yaml` | 50 位 P 主配置 |
| `configs/training_catalog_additions.yaml` | 训练目录补充 |
| `configs/dev_holdout_catalog.yaml` | dev holdout 配置 |
| `configs/frozen_test_catalog.yaml` | final frozen 配置 |
| `src/vocaptest/models/calibrated_stacking.py` | P4 calibrated stacking 模型类 |
| `scripts/35_train_p4_calibrated_stacking.py` | 训练部署 artifact |
| `data/processed/models/p4_calibrated_stacking.pkl` | 当前部署模型，未纳入 git，由部署脚本同步 |
| `data/processed/evaluations/p4_calibrated_stacking_deploy.json` | 部署模型评估结果 |

## 复现实验

P4 搜索相关脚本按时间顺序排列：

```powershell
python scripts/27_run_p4_broad_model_search.py
python scripts/28_validate_p4_broad_candidates.py
python scripts/29_run_p4_concat_pooling_search.py
python scripts/30_run_p4_similarity_search.py
python scripts/31_run_p4_stacking_search.py
python scripts/32_run_p4_projection_head_search.py
python scripts/33_run_p4_cv_selected_stacking.py
python scripts/34_run_p4_calibrated_stacking.py
python scripts/35_train_p4_calibrated_stacking.py
```

历史数据准备和 MERT 缓存重建仍使用：

```powershell
python scripts/16_expand_training_catalog.py --ffmpeg-location <ffmpeg目录>
python scripts/09_rebuild_p1_layer_embeddings.py
python scripts/21_train_p3_layer_fusion.py
```

新增或替换 P 主时，请按 [Catalog Selection Protocol](docs/CATALOG_SELECTION_PROTOCOL.md) 执行：VocaDB 为主证据，支持 YouTube 和 niconico Original PV，训练/dev/final 按 VocaDB song id 与媒体 source key 双重隔离，不能为了凑数量混入来源不清或风格归因不稳的曲目。
