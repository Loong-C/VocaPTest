# Voca-like — 测测你的曲风最像哪位 P 主

一个娱乐向的 Vocaloid Producer 风格相似度系统。上传一段音乐，系统在预先构建的 P 主参考库中寻找听感最接近的 Producer，输出 Top-K 相似结果。

## 技术路线

- **音频特征提取**：MERT-v1-95M（主力）/ MuQ（备选）
- **相似度检索**：余弦相似度 + KMeans 多原型 Profile
- **后端**：FastAPI + Uvicorn
- **前端**：Next.js / Vite + React
- **数据源**：VocaDB API → yt-dlp 音频下载

## 开发阶段

| 阶段 | 内容 | 状态 |
|------|------|:----:|
| 0 | 项目初始化 + 设计文档 | ✅ |
| 1 | 项目脚手架（目录/配置/依赖） | ✅ |
| 2 | 核心包结构 + utils + schemas | ✅ |
| 3 | 数据管线（VocaDB/下载/清洗） | ⬜ |
| 4 | 音频预处理 + 切片 | ⬜ |
| 5 | 模型加载 + Embedding 提取 | ⬜ |
| 6 | 检索系统（Profile/相似度/搜索） | ⬜ |
| 7 | FastAPI 后端服务 | ⬜ |
| 8 | 脚本 + CLI + 最终整理 | ⬜ |

## 项目结构

```text
voca-like/
  configs/         # YAML 配置文件
  data/            # 原始/中间/处理后数据
  external/        # 外部参考仓库
  src/vpstyle/     # Python 核心包
  web/             # 前端
  scripts/         # 一键流程脚本
  notebooks/       # 探索分析
  tests/           # 测试
  docs/            # 文档
```

## 环境要求

- Python 3.13+
- PyTorch + CUDA（推荐 GPU）
- 详见 `requirements.txt`

## 快速开始

```bash
# 创建环境
conda create -n vpstyle python=3.13 -y
conda activate vpstyle

# 安装依赖
pip install -r requirements.txt

# 初始化目录
python scripts/00_init_dirs.py
```

详细设计文档见 [agent.md](agent.md)。
