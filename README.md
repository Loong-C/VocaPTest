# VocaP Test — 测测你的曲风最像哪位 P 主

一个娱乐向的 Vocaloid Producer 风格相似度系统。上传一段音乐，系统在预先构建的 P 主参考库中寻找听感最接近的 Producer，输出 Top-K 相似结果。

## 已完成工作总览

### 当前可信数据管线

| 步骤 | 内容 | 结果 |
|------|------|------|
| 原始数据 | YouTube 管线收集 | 211 首 |
| 清洗 | 排除错误实体、翻唱、游戏版、合作污染和重复作品 | **174 首 canonical 歌曲** |
| 切片 | 24kHz、20s 窗口、10s hop、均匀覆盖、最多 12 段 | **2086 段** |
| 嵌入 | 一次前向缓存 MERT-v1-95M 全部 13 层 | `13 × 768` / 段 |
| 模型 | 第 6 层歌曲均值 + 等先验 Shrinkage LDA | 18 类 |
| 置信度 | OOF logits temperature scaling + 拒识阈值 | 95% 接受精度目标 |

### 覆盖的 18 位 P 主

| P 主 | 别名 | 歌曲数 | 分段数 |
|------|------|:------:|:------:|
| wowaka | 現実逃避P | 5 | 60 |
| kemu | — | 10 | 120 |
| Neru | 押入れP | 5 | 60 |
| DECO*27 | — | 12 | 144 |
| ピノキオピー | — | 11 | 131 |
| Mitchie M | — | 8 | 96 |
| じん | — | 11 | 132 |
| Orangestar | — | 10 | 120 |
| cosMo@暴走P | — | 11 | 132 |
| ハチ | — | 10 | 120 |
| 40mP | — | 10 | 120 |
| ナユタン星人 | — | 9 | 108 |
| かいりきベア | — | 10 | 120 |
| Kanaria | — | 7 | 83 |
| Chinozo | — | 11 | 132 |
| 稲葉曇 | — | 12 | 144 |
| MIMI | — | 12 | 144 |
| MARETU | — | 10 | 120 |

### P1 评估结果

使用 5 次重复 5 折 `StratifiedGroupKFold`，按 `work_id` 分组：

| 指标 | P0 最后一层 | P1 第 6 层 |
|------|:-----------:|:----------:|
| Top-1 | 62.47% ± 1.99% | **87.47% ± 1.25%** |
| Top-3 | 79.48% ± 2.25% | **95.63% ± 1.38%** |
| Macro-F1 | 59.73% ± 2.40% | **85.70% ± 1.67%** |
| MRR | 72.94% ± 1.49% | **91.72% ± 0.70%** |

校准阈值在 OOF 上覆盖 77.59% 的样本，接受样本精度为 95.11%。详细消融见
[P1 实施报告](docs/P1_IMPLEMENTATION_REPORT.md)。

### API 服务

FastAPI 后端已部署，支持以下端点：

| 端点 | 方法 | 说明 |
|------|:----:|------|
| `/health` | GET | 健康检查 |
| `/api/producers` | GET | 获取所有 P 主列表 |
| `/api/analyze` | POST | 上传音频 → 返回 Top-K P 主匹配结果 |

```bash
# 启动 API 服务
python scripts/06_run_api.py
# → http://localhost:8000 (Swagger UI: /docs)

# 启动前端开发服务器
cd web && npm run dev
# → http://localhost:5173 (Vite 代理 /api 到 :8000)
```

### 关键文件位置

| 文件 | 说明 |
|------|------|
| `data/processed/profiles.pkl` | 全量 Profile（18 P 主 × 5 原型） |
| `data/processed/models/p1_selected_layer_lda.pkl` | 当前默认模型，本地生成且不提交 |
| `data/processed/evaluations/p1_selected_layer.json` | 最终分组 CV 与校准结果 |
| `data/processed/evaluations/p1_feature_ablations.json` | 多统计和 MIR 消融 |
| `data/processed/curated/mert_95_p1/segments.jsonl` | P1 多层 embedding 清单 |
| `data/processed/song_name_mapping.csv` | YouTube ID → 歌曲名映射（211 首） |
| `data/interim/youtube_songs.jsonl` | 下载元数据 |
| `data/interim/segments.jsonl` | 分段清单 |
| `configs/producers.yaml` | P 主配置 |

## 技术路线

- **音频特征提取**：MERT-v1-95M 第 6 层
- **分类**：歌曲均值 + Shrinkage LDA
- **置信度**：temperature scaling、margin、归一化熵和拒识阈值
- **后端**：FastAPI + Uvicorn
- **前端**：Vite + React + Tailwind CSS + TypeScript
- **数据源**：YouTube 搜索 → yt-dlp 音频下载

## 开发阶段

| 阶段 | 内容 | 状态 |
|------|------|:----:|
| 0 | 项目初始化 + 设计文档 | ✅ |
| 1 | 项目脚手架（目录/配置/依赖） | ✅ |
| 2 | 核心包结构 + utils + schemas | ✅ |
| 3 | 数据管线（YouTube 下载/清洗/验证） | ✅ |
| 4 | 音频预处理 + 切片 | ✅ |
| 5 | 模型加载 + Embedding 提取 | ✅ |
| 6 | 检索系统（Profile/相似度/搜索） | ✅ |
| 7 | FastAPI 后端服务 | ✅ |
| 8 | 脚本 + 训练/测试划分 + 评估 | ✅ |
| 9 | 前端开发 | ✅ |

## 项目结构

```text
VocaP Test/
  configs/         # YAML 配置文件
  data/            # 原始/中间/处理后数据
  external/        # 外部参考仓库
  src/vocaptest/   # Python 核心包
  web/             # 前端 (Vite + React + Tailwind)
  scripts/         # 一键流程脚本
  notebooks/       # 探索分析
  tests/           # 测试
  docs/            # 文档
```

## 已知问题

1. 尚无永久冻结的外部测试集；当前层选择仍可能对 174 首数据产生选择偏差。
2. wowaka 和 Neru 各只有 5 首 canonical 作品。
3. 校准拒识基于库内 OOF，不等同于完整的开放集识别保证。
4. 歌声库、上传频道和母带风格仍可能成为作曲家标签的代理变量。

## 环境要求

- Python 3.12+
- PyTorch + CUDA（推荐 GPU，CPU 推理也可用但较慢）
- 详见 `requirements.txt`

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 首次生成 P1 缓存和模型
python scripts/09_rebuild_p1_layer_embeddings.py
python scripts/13_train_p1_selected_layer.py

# 启动 API
python scripts/06_run_api.py
```

详细设计文档见 [agent.md](agent.md)。
