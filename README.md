# VocaP Test — 测测你的曲风最像哪位 P 主

一个娱乐向的 Vocaloid Producer 风格相似度系统。上传一段音乐，系统在预先构建的 P 主参考库中寻找听感最接近的 Producer，输出 Top-K 相似结果。

## 已完成工作总览

### 数据管线

| 步骤 | 内容 | 结果 |
|------|------|------|
| 下载 | YouTube 搜索 + yt-dlp 下载 18 位 P 主各 12 首 | **211 首** MP3（5 首下载失败） |
| 验证 | 跨 P 主去重、合作曲目标注、数据集质量检查 | 0 重复，4 首合作曲已标注 |
| 切片 | 30s 无重叠 WAV、24kHz 单声道、RMS 能量过滤 | **1466 段** |
| 嵌入 | MERT-v1-95M 提取 768 维特征（CUDA 加速） | **1466 条**嵌入（~2.7 min） |
| Profile | KMeans 聚类 5 原型 / P 主 | 18 个 Profile（pickle） |
| 评估 | 按歌曲划分训练/测试集，诚实评估 | 见下方评估结果 |

### 覆盖的 18 位 P 主

| P 主 | 别名 | 歌曲数 | 分段数 |
|------|------|:------:|:------:|
| wowaka | 現実逃避P | 12 | ~84 |
| kemu | — | 11 | ~77 |
| Neru | 押入れP | 12 | ~84 |
| DECO*27 | — | 12 | ~84 |
| ピノキオピー | — | 12 | ~84 |
| Mitchie M | — | 12 | ~84 |
| じん | — | 12 | ~84 |
| Orangestar | — | 12 | ~84 |
| cosMo@暴走P | — | 11 | ~77 |
| ハチ | — | 12 | ~84 |
| 40mP | — | 11 | ~77 |
| ナユタン星人 | — | 12 | ~84 |
| かいりきベア | — | 12 | ~84 |
| Kanaria | — | 11 | ~77 |
| Chinozo | — | 12 | ~84 |
| 稲葉曇 | — | 12 | ~84 |
| MIMI | — | 12 | ~84 |
| MARETU | — | 12 | ~84 |

### 评估结果（诚实 train/test split）

使用 **按歌曲分层划分** 避免数据泄漏（同一首歌的不同分段不会同时出现在训练集和测试集）：

| 指标 | 训练集自检 | 测试集（36 首未见歌曲） |
|------|:----------:|:------------------------:|
| Top-1 准确率 | 97.1% (170/175) | **30.6%** (11/36) |
| Top-3 准确率 | 98.9% (173/175) | **63.9%** (23/36) |

> 随机猜测基线为 1/18 ≈ 5.6%，模型在未见歌曲上的 Top-1 约为基线的 **5.5 倍**，Top-3 约为基线的 **11.5 倍**。
>
> ⚠️ **重要教训**：初次评估时未做歌曲级划分，同一首歌的不同分段泄漏到训练集导致虚高的 95.7% Top-1。修正后取得以上诚实结果。

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
| `data/processed/train_profiles.pkl` | 仅训练集 Profile |
| `data/processed/song_name_mapping.csv` | YouTube ID → 歌曲名映射（211 首） |
| `data/interim/youtube_songs.jsonl` | 下载元数据 |
| `data/interim/segments.jsonl` | 分段清单 |
| `configs/producers.yaml` | P 主配置 |

## 技术路线

- **音频特征提取**：MERT-v1-95M（主力）/ MuQ（备选）
- **相似度检索**：余弦相似度 + KMeans 多原型 Profile
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

1. **下载失败 5 首**：kemu《インビジブル》(403)、cosMo《ダイジョブですか》(403)、ハチ《リンネ》(年龄限制)、40mP 1 首 (403)、Kanaria 1 首（Premieres 在 3 小时后）
2. **合作曲目**：4 首跨 P 主合作曲已标注，但在 Profile 构建中按主要 P 主归类
3. **NumPy 2.x 兼容性**：`librosa.load()` 在 NumPy 2.x 下与 numba 冲突，已全部替换为 `soundfile.read()` + `scipy.signal.resample()`

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

# 启动 API（无需重新训练）
python scripts/06_run_api.py
```

详细设计文档见 [agent.md](agent.md)。
