# VocaP Test — 测测你的曲风最像哪位 P 主

一个娱乐向的 Vocaloid Producer 风格相似度系统。上传一段音乐，系统在预先构建的 P 主参考库中寻找听感最接近的 Producer，输出 Top-K 相似结果。

## 已完成工作总览

### 当前可信数据管线

| 步骤 | 内容 | 结果 |
|------|------|------|
| 原始数据 | YouTube 管线收集 + VocaDB 人工核验扩充 | 276 条候选记录 |
| 清洗 | 排除错误实体、翻唱、游戏版、合作污染和重复作品 | **239 首 canonical 歌曲** |
| 切片 | 24kHz、20s 窗口、10s hop、均匀覆盖、最多 12 段 | **2866 段** |
| 嵌入 | 一次前向缓存 MERT-v1-95M 全部 13 层 | `13 × 768` / 段 |
| 模型 | 第 6 层歌曲均值 + 等先验 Shrinkage LDA | 20 类 |
| 置信度 | OOF logits temperature scaling + 拒识阈值 | 95% 接受精度目标 |

### 覆盖的 20 位 P 主

| P 主 | 别名 | 歌曲数 | 分段数 |
|------|------|:------:|:------:|
| wowaka | 現実逃避P | 10 | 120 |
| kemu | — | 12 | 144 |
| Neru | 押入れP | 10 | 120 |
| DECO*27 | — | 14 | 168 |
| ピノキオピー | — | 13 | 155 |
| Mitchie M | — | 10 | 120 |
| じん | — | 13 | 156 |
| Orangestar | — | 12 | 144 |
| cosMo@暴走P | — | 13 | 156 |
| ハチ | — | 12 | 144 |
| 40mP | — | 12 | 144 |
| ナユタン星人 | — | 11 | 132 |
| かいりきベア | — | 12 | 144 |
| Kanaria | — | 10 | 119 |
| Chinozo | — | 13 | 156 |
| 稲葉曇 | — | 14 | 168 |
| MIMI | — | 14 | 168 |
| MARETU | — | 12 | 144 |
| n-buna | ナブナ | 11 | 132 |
| Ayase | — | 11 | 132 |

### P1 评估结果

使用 5 次重复 5 折 `StratifiedGroupKFold`，按 `work_id` 分组：

| 指标 | 20 类 / 192 首 | 当前 20 类 / 239 首 |
|------|:--------------:|:-------------------:|
| Top-1 | 86.67% ± 1.36% | **88.20% ± 0.55%** |
| Top-3 | 96.35% ± 0.74% | **95.56% ± 0.56%** |
| Macro-F1 | 85.64% ± 1.52% | **88.48% ± 0.55%** |
| MRR | 91.62% ± 0.76% | **92.30% ± 0.29%** |

补齐低样本类别后，wowaka 的 OOF F1 从约 0.73 升至 0.89，Neru 从约
0.62 升至 0.75。校准阈值在 OOF 上覆盖 83.51% 的样本，接受样本精度为
95.09%。这说明模型已适合当前 20 类库内的娱乐性匹配，但没有冻结库外测试集，
不能把 88.20% 解读为任意歌曲上的作曲家鉴定准确率。详细消融见
[P1 实施报告](docs/P1_IMPLEMENTATION_REPORT.md)。

### API 服务

FastAPI 后端已部署，支持以下端点：

| 端点 | 方法 | 说明 |
|------|:----:|------|
| `/health` | GET | 健康检查 |
| `/api/producers` | GET | 获取所有 P 主列表 |
| `/api/producers/{slug}` | GET | 获取头像、别名和实际训练曲目 |
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
| `data/processed/profiles.pkl` | 旧检索原型文件；当前 API 使用 P1 LDA 模型 |
| `data/processed/models/p1_selected_layer_lda.pkl` | 当前默认模型，本地生成且不提交 |
| `data/processed/evaluations/p1_selected_layer.json` | 最终分组 CV 与校准结果 |
| `data/processed/evaluations/p1_feature_ablations.json` | 多统计和 MIR 消融 |
| `data/processed/curated/mert_95_p1/segments.jsonl` | P1 多层 embedding 清单 |
| `data/processed/song_name_mapping.csv` | 初始 YouTube 数据的歌曲名映射 |
| `data/interim/youtube_songs.jsonl` | 下载元数据 |
| `data/interim/segments.jsonl` | 分段清单 |
| `configs/producers.yaml` | P 主配置 |
| `configs/training_catalog_additions.yaml` | 人工核验的新增歌曲与频道白名单 |

## 技术路线

- **音频特征提取**：MERT-v1-95M 第 6 层
- **分类**：歌曲均值 + Shrinkage LDA
- **置信度**：temperature scaling、margin、归一化熵和拒识阈值
- **后端**：FastAPI + Uvicorn
- **前端**：Vite + React + Tailwind CSS + TypeScript
- **数据源**：YouTube 音频 + VocaDB Original 条目核验 + 官方频道 ID 白名单
- **图鉴**：本地化 VocaDB 头像、别名和可点击的实际训练曲目
- **低置信 UI**：不再显示黄色英文警告框，改为非确定性的中文参考结果

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

1. 尚无永久冻结的外部测试集；层选择和拒识阈值仍可能对当前数据产生选择偏差。
2. 校准拒识基于库内 OOF，不等同于完整的开放集识别保证。
3. 歌声库、上传频道和母带风格仍可能成为作曲家标签的代理变量。
4. じん当前 OOF F1 为 58.75%，主要问题是类别异质性和与其他摇滚作者的边界，
   继续盲目加歌不一定有效，应先审计作品口径。
5. Neru recall 已从 48% 升至 68%，但仍有 28% 的 OOF 预测落到 じん。

## 能否继续增加 P 主

可以。当前 RTX 4060 Ti 8GB、31.7GB 内存下，MERT-v1-95M batch size 4
运行稳定；本次新增 47 首的 13 层缓存约 2 分钟，LDA 训练约 23 秒。真正的
限制是数据质量和评估设计：

- 每位新 P 主建议至少准备 10 首互不重复的 canonical 作品，最好 12–15 首；
- 另留 2–3 首从不参与层选择、训练和阈值选择的冻结外部测试作品；
- 每次增加约 3–5 位 P 主后重跑分组 CV、逐类 F1、混淆矩阵和校准；
- 优先选择创作身份明确、合作污染少、作品时期有覆盖的 P 主；
- 不建议为了扩大名单使用翻唱、游戏剪辑、现场版或同一作品的多个上传版本。

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
python scripts/16_expand_training_catalog.py --ffmpeg-location <ffmpeg目录>
python scripts/09_rebuild_p1_layer_embeddings.py
python scripts/13_train_p1_selected_layer.py

# 刷新本地头像
python scripts/15_sync_producer_avatars.py

# 启动 API
python scripts/06_run_api.py
```

详细设计文档见 [agent.md](agent.md)。
