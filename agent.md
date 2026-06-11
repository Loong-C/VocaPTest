# 测测你的曲风最像哪位 P 主

## 项目目标

本项目的目标是构建一个非严肃、娱乐向的音乐相似度系统。用户上传一段音乐或一首完整歌曲后，系统会在预先构建的 Vocaloid Producer 参考库中寻找最相近的 P 主，并输出 Top-K 相似结果，例如“最像 wowaka、kemu、Neru 这一带”，同时给出若干可解释的风格描述，例如高速、电子摇滚、钢琴透明感、暗黑叙事、kawaii、爆速、电波、调声拟真等。

项目第一阶段不追求严肃的“作者识别”，也不声称模型真正理解了作曲家的音乐学风格。它更接近一个音频 embedding 检索系统：把每位 P 主的代表作映射成向量空间中的若干风格原型，再把用户上传的歌曲映射到同一空间中，计算相似度。

因此，第一版的核心目标不是“准确判断这首歌是谁写的”，而是“在参考库中找到听感上比较接近的 P 主原型，并以有趣、可解释的方式展示结果”。

## MVP 范围

第一版建议只覆盖 10 到 20 位风格差异明显的 P 主。不要一开始做 100 位以上，因为数据清洗、标签歧义、合作曲、Remix、翻唱、音源差异都会迅速放大难度。

第一版推荐选择风格轮廓较鲜明、代表作较多、听众印象较稳定的 P 主，推荐名单如下（实际进入数据集的名单应以数据可获取性和清洗质量为准）：

- wowaka（現実逃避P）、kemu、Neru（押入れP）、DECO*27、ピノキオピー、Mitchie M、じん、Orangestar、cosMo@暴走P、ハチ、40mP、ナユタン星人、かいりきベア、Kanaria、Chinozo、稲葉曇、MIMI、MARETU

每位 P 主第一版建议收集 20 到 50 首歌曲。每首歌切成多个 10 到 30 秒片段，每个片段提取 embedding。假设选择 20 位 P 主，每位 30 首歌，每首歌切 8 个片段，那么可以得到约 4800 个片段样本。这个规模不适合从零训练大模型，但足够用于 frozen backbone embedding、原型相似度检索、kNN、线性分类器或小型 MLP probe。

第一版输出建议采用 Top-5，而不是 Top-1。"曲风像谁"本身不是单标签分类问题。输出形式只包含 P 主名称和相似度分数，不做风格区域分类、原因解释等附加输出。

## 总体技术路线

第一阶段采用“预训练音乐模型提特征 + 相似度检索”的路线。

系统流程如下：

用户上传音频。后端将音频转为统一采样率、单声道或立体声规范格式。然后进行静音检测和片段切分。每个片段输入音乐基座模型，例如 MERT、MuQ 或 CLAP-Music，得到向量 embedding。所有片段 embedding 经过平均池化、加权池化或 top-k pooling，得到整首歌的表示。系统再将该表示与参考库中的 P 主原型向量比较，计算余弦相似度，输出最接近的几个 P 主。

参考库的构建流程如下：

先用 VocaDB API 或人工方式建立 P 主与歌曲列表。再获取歌曲音频文件。然后对每首参考歌曲进行预处理、切片、embedding 提取。最后按 P 主聚合 embedding，形成每位 P 主的一个或多个 centroid。每位 P 主不要只保留一个均值，因为很多 P 主有多个时期和多个风格侧面。更推荐每位 P 主通过 KMeans 保留 3 到 8 个风格原型。

第一版可以完全不训练，只做 embedding + centroid 检索。第二版再加入 logistic regression、linear SVM 或 MLP probe。第三版再考虑微调 MERT/MuQ 的最后几层，或者做 LoRA/adapter 式微调。

## 推荐仓库结构

建议仓库从一开始就分清“原始音频”“元数据”“中间缓存”“模型代码”“服务端”“前端”“实验记录”。不要把所有脚本堆在根目录。

```tex
VocaP Test/
  README.md
  pyproject.toml
  requirements.txt
  .env.example
  .gitignore

  configs/
    default.yaml
    dataset.yaml
    model_mert.yaml
    model_muq.yaml
    retrieval.yaml
    api.yaml

  data/
    raw/
      audio/
        producers/
          wowaka/
          kemu/
          neru/
        uploads/
      metadata/
        vocadb_artists.jsonl
        vocadb_songs.jsonl
        song_links.jsonl
    interim/
      wav_24k/
      segments/
      vad_reports/
    processed/
      embeddings/
        mert_95/
        mert_330/
        muq/
      producer_profiles/
        profiles_mert_95.pkl
        profiles_muq.pkl
      splits/
        train_song_ids.txt
        val_song_ids.txt
        test_song_ids.txt

  external/
    FM-music-tagging/
    MERT/
    MuQ/

  src/
    vocaptest/
      __init__.py

      data/
        vocadb_client.py
        metadata_schema.py
        build_song_index.py
        download_audio.py
        validate_dataset.py

      audio/
        preprocess.py
        segment.py
        loudness.py
        source_separation.py
        mir_features.py

      models/
        base.py
        mert_embedder.py
        muq_embedder.py
        clap_embedder.py
        pooling.py

      features/
        extract_embeddings.py
        build_feature_store.py
        feature_store.py

      retrieval/
        build_profiles.py
        similarity.py
        search.py
        calibrate_scores.py

      training/
        train_probe.py
        train_metric.py
        evaluate.py
        split.py

      api/
        main.py
        schemas.py
        dependencies.py
        routes_upload.py
        routes_search.py
        routes_metadata.py

      utils/
        paths.py
        logging.py
        config.py
        hashing.py

  web/
    package.json
    src/
      app/
      components/
      lib/
      styles/

  scripts/
    00_init_dirs.py
    01_fetch_vocadb_metadata.py
    02_prepare_audio.py
    03_extract_embeddings.py
    04_build_profiles.py
    05_evaluate_retrieval.py
    06_run_api.py

  notebooks/
    01_embedding_visualization.ipynb
    02_error_analysis.ipynb
    03_producer_clusters.ipynb

  tests/
    test_audio_preprocess.py
    test_segment.py
    test_similarity.py
    test_api.py

  docs/
    dataset_building.md
    model_notes.md
    evaluation_protocol.md
    product_design.md
```

这个结构里，data/raw 保存原始音频和原始元数据，data/interim 保存统一采样率后的 wav、切片、静音检测报告，data/processed 保存 embedding 和 P 主 profile。external 保存参考或克隆来的外部仓库，但不建议直接把外部仓库代码深度耦合到主项目。src/vocaptest 是你的正式 Python 包，所有稳定代码都放在这里。scripts 是面向流程的一键脚本。notebooks 用于观察、可视化和错误分析，不作为正式生产逻辑。

## 环境准备

推荐使用 Python 3.10+。当前开发环境使用 Python 3.13，已验证 torch、numba、faiss-cpu 等关键依赖均有 cp313 预编译 wheel，可以直接使用 3.13。第一版建议优先使用 PyTorch、transformers、torchaudio、librosa、scikit-learn、fastapi、uvicorn、pydantic、numpy、pandas、faiss-cpu 或 hnswlib。

基础依赖可以先写成：

```txt
torch
torchaudio
transformers
accelerate
librosa
soundfile
numpy
pandas
scikit-learn
scipy
tqdm
pyyaml
pydantic
fastapi
uvicorn
python-multipart
faiss-cpu
hnswlib
matplotlib
umap-learn
requests
httpx
yt-dlp
python-dotenv
```

如果使用 MuQ，需要额外安装：

```bash
pip install muq
```

如果使用 MERT，一般可以通过 Hugging Face transformers 加载，优先不要从官方 fairseq 训练代码开始。官方 MERT 仓库更适合作为论文与模型来源参考，而不是第一版项目直接依赖。第一版只需要 Hugging Face 上的 MERT 权重即可。

推荐创建环境：

```bash
conda create -n vocaptest python=3.13 -y
conda activate vocaptest
pip install -r requirements.txt
```

如果有 NVIDIA GPU，先根据 CUDA 版本安装对应 PyTorch。没有 GPU 也可以运行第一版，但 embedding 提取会慢很多。MVP 阶段可以先离线批量提取参考库 embedding，线上只处理用户上传音频，因此 CPU 也能做 demo，只是等待时间较长。

## 配置文件设计

configs/default.yaml 应该负责全局路径、采样率、片段长度、模型选择、检索参数。

示例：

```yaml
project:
  name: vocaloid-producer-style
  seed: 42

paths:
  data_root: data
  raw_audio_dir: data/raw/audio/producers
  upload_dir: data/raw/audio/uploads
  metadata_dir: data/raw/metadata
  interim_dir: data/interim
  embedding_dir: data/processed/embeddings
  profile_dir: data/processed/producer_profiles

audio:
  sample_rate: 24000
  mono: true
  segment_seconds: 20
  hop_seconds: 10
  min_rms_db: -45
  max_segments_per_song: 12
  trim_silence: true

model:
  backend: mert_95
  device: cuda
  batch_size: 8
  pooling: mean

retrieval:
  metric: cosine
  producer_clusters: 5
  top_k: 5
  segment_top_ratio: 0.4

api:
  host: 0.0.0.0
  port: 8000
  max_upload_mb: 50
```

configs/model_mert.yaml 可以写：

```yaml
model:
  backend: mert_95
  hf_name: m-a-p/MERT-v1-95M
  sample_rate: 24000
  layer_strategy: mean_last_hidden
```

configs/model_muq.yaml 可以写：

```yaml
model:
  backend: muq
  hf_name: OpenMuQ/MuQ-large-msd-iter
  sample_rate: 24000
  use_fp32: true
```

注意 MuQ 官方说明输入严格要求 24kHz，并建议推理时使用 fp32 避免潜在 NaN 问题。因此项目内部最好统一使用 24kHz。即使 MERT 也可接受相近配置，统一采样率会减少工程复杂度。

## 数据来源与数据集构建

第一版数据集由三部分组成：P 主列表、歌曲元数据、音频文件。

P 主列表建议人工确定，不要完全自动抓取。因为项目目标是娱乐向“风格像谁”，第一版更应该选择有代表性、风格轮廓明显、歌曲数量足够、用户熟悉度较高的 P 主。

歌曲元数据可以从 VocaDB 获取。VocaDB 提供 artist、album、song 等信息查询，可以用于建立 P 主与歌曲列表。

**VocaDB → yt-dlp 下载链路已确认可行**：通过 VocaDB API `/api/songs?artistId=X&fields=PVs` 获取歌曲的 PV 列表，过滤 `service=Youtube` 且 `pvType=Original` 的条目，提取 `url` 字段（格式如 `https://youtu.be/vnw8zURAxkU`），即可用 yt-dlp 下载音频。你需要记录的信息至少包括 producer_id、producer_name、song_id、song_name、publish_date、vocalists、artist_roles、pv_links、tags、song_type、is_cover、is_remix、is_collaboration、source_url。

音频文件可以来自 YouTube、Niconico、Bilibili、个人已有音频文件或其他平台。暂且不考虑版权时，可以用 yt-dlp 统一下载音频。但项目结构仍然要设计成可以替换音频来源：metadata 中只保存 source_url 和 local_audio_path，不把下载逻辑写死在模型代码里。

数据构建流程建议如下：

首先建立 producers.yaml，手动写入第一批 P 主。

```yaml
producers:
  - slug: wowaka
    display_name: wowaka
    vocadb_artist_id: null
    aliases: ["現実逃避P"]
  - slug: kemu
    display_name: kemu
    vocadb_artist_id: null
    aliases: []
  - slug: neru
    display_name: Neru
    vocadb_artist_id: null
    aliases: ["押入れP"]
```

然后写脚本 01_fetch_vocadb_metadata.py，通过 VocaDB API 搜索 artist，并拉取对应歌曲。因为同名、别名、合作名义可能混淆，所以不要让脚本自动确认所有匹配结果。更推荐先生成候选 artist 表，然后人工确认 vocadb_artist_id。

元数据保存为 jsonl，每行一首歌：

```json
{"producer_slug":"wowaka","song_id":"vocadb_123","title":"example","publish_date":"2011-01-01","pv_links":[{"service":"Youtube","url":"..."}],"vocalists":["Hatsune Miku"],"tags":["VOCAROCK"],"is_cover":false,"is_remix":false,"is_collaboration":false,"local_audio_path":null}
```

下载脚本 02_prepare_audio.py 可以读取 song_links.jsonl，对有 source_url 但没有 local_audio_path 的记录调用 yt-dlp。建议所有文件名使用 hash，而不是直接使用歌名，避免日文、特殊符号、重复标题导致路径问题。

```bash
yt-dlp -x --audio-format wav --audio-quality 0 -o "data/raw/audio/producers/%(id)s.%(ext)s" "<url>"
```

下载后立刻生成 audio_manifest.jsonl，记录文件 hash、时长、采样率、声道数、来源 URL、对应 producer_slug、song_id。

## 数据清洗规则

第一版一定要做最小清洗，否则模型结果会非常不稳定。

**自动清洗**（通过 VocaDB 元数据字段判断）：
- 根据 `songType` 字段排除 Cover、Remix、Instrumental、Live、Other 等非 Original 类型
- 根据 `artists` 中 `categories` 包含 "Producer" 的人数判断是否为合作曲：如果超过 1 位 Producer，标记为 `collaboration` 并排除
- 根据 `lengthSeconds` 排除过短（<60s）的 preview/short version

**人工确认**（自动过滤后不确定的边界情况）：
- 同名歌曲的不同投稿版本（同一 song_id 只保留一个 PV）
- 别名、合作名义混淆导致同一首歌被多次收录
- 每位 P 主尽量平衡歌曲数量，目标每人 20~50 首

可以给每首歌一个 status 字段：

```json
{"song_id":"...","status":"accepted","reason":null}
{"song_id":"...","status":"rejected","reason":"cover"}
{"song_id":"...","status":"rejected","reason":"collaboration"}
{"song_id":"...","status":"rejected","reason":"short_preview"}
{"song_id":"...","status":"pending_review","reason":"multiple_producers_need_check"}
```

第一版不需要完美，但需要可追踪。每次删除一首歌，最好记录原因。以后模型出错时可以回看数据清洗是否合理。

自动清洗脚本遇到不确定情况（如多个 Producer 但 roles 不明确、songType 标记矛盾等）应标记为 `pending_review` 并暂停等待人工确认。

## 音频预处理

音频预处理应该统一在 src/vocaptest/audio/preprocess.py 中实现。

预处理步骤建议包括：读取音频，转 mono，重采样到 24kHz，归一化响度或峰值，裁剪过长静音，保存到 data/interim/wav_24k。

伪代码如下：

```python
import librosa
import soundfile as sf
from pathlib import Path

def load_audio(path: str, sr: int = 24000, mono: bool = True):
    wav, _ = librosa.load(path, sr=sr, mono=mono)
    return wav

def normalize_peak(wav, peak=0.95):
    max_abs = abs(wav).max()
    if max_abs < 1e-8:
        return wav
    return wav / max_abs * peak

def save_wav(wav, path: str, sr: int = 24000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr)
```

切片逻辑放在 src/vocaptest/audio/segment.py。建议第一版使用固定长度滑窗，例如 segment_seconds=20，hop_seconds=10。不要只取歌曲中间 30 秒，因为很多 Vocaloid 曲的风格特征可能出现在副歌、高速段、drop、间奏、调声密集段。也不要切太短，10 秒以下可能缺少结构信息。20 到 30 秒是比较合理的起点。

片段筛选可以用 RMS 能量过滤静音和过弱片段。第一版不用复杂 VAD，因为音乐不是语音。可以计算每个片段的 RMS dB，低于阈值就丢弃。每首歌最多保留 8 到 12 个片段，以免长歌拥有过多权重。

## 模型加载方案

第一版建议实现统一 Embedder 接口。无论后端是 MERT、MuQ 还是 CLAP，都暴露相同方法：

```python
class AudioEmbedder:
    def __init__(self, config):
        pass

    def embed_file(self, wav_path: str):
        pass

    def embed_batch(self, wavs):
        pass
```

这样后续替换模型不会影响检索系统。

### MERT 加载

MERT 第一版建议作为主力 baseline。使用 Hugging Face transformers 加载，不直接训练官方 fairseq 代码。

示例代码：

```python
import torch
import librosa
from transformers import AutoModel, AutoProcessor

class MERTEmbedder:
    def __init__(self, model_name="m-a-p/MERT-v1-95M", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_file(self, wav_path: str, sr: int = 24000):
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        inputs = self.processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]
        emb = hidden.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy()
```

实际代码中需要根据所选 MERT 权重的 processor 行为调整。第一版可以先把 MERT-v1-95M 跑通，再尝试 MERT-v1-330M。95M 速度更快，适合开发；330M 可能效果更好，但显存、速度成本更高。

### MuQ 加载

MuQ 适合作为第二个强 baseline。它有 pip 包，官方示例非常直接。注意 24kHz 输入和 fp32。

示例代码：

```python
import torch
import librosa
from muq import MuQ

class MuQEmbedder:
    def __init__(self, model_name="OpenMuQ/MuQ-large-msd-iter", device="cuda"):
        self.device = device
        self.model = MuQ.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_file(self, wav_path: str, sr: int = 24000):
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        wavs = torch.tensor(wav).float().unsqueeze(0).to(self.device)

        output = self.model(wavs)
        if isinstance(output, dict):
            features = output.get("last_hidden_state", None)
            if features is None:
                features = output.get("x", None)
        else:
            features = output

        if features.dim() == 3:
            emb = features.mean(dim=1)
        else:
            emb = features

        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy()
```

这里的 output 字段需要以实际 MuQ 版本为准。第一次接入时不要急着封装复杂逻辑，先在 notebook 中打印 output 类型和 shape，再写正式 wrapper。

第一版主检索用 MERT 或 MuQ。CLAP / MuQ-MuLan 作为备选方案，如果后续需要文本相关功能再考虑接入。

## Embedding 缓存设计

embedding 提取很耗时，所以必须缓存。不要每次启动 API 都重新跑模型。

建议每个片段保存一个 .npy 文件，并在 manifest 中记录其来源。

data/processed/embeddings/mert_95/segments.jsonl：

```json
{"segment_id":"abc123","producer_slug":"wowaka","song_id":"vocadb_123","segment_path":"data/interim/segments/abc123.wav","embedding_path":"data/processed/embeddings/mert_95/abc123.npy","start_sec":30.0,"end_sec":50.0}
```

extract_embeddings.py 流程如下：

读取 segment manifest。检查 embedding 文件是否存在。不存在则加载模型提取。保存 npy。最后写入 embeddings manifest。

脚本命令示例：

```bash
python scripts/03_extract_embeddings.py \
  --config configs/model_mert.yaml \
  --segments data/interim/segments/segments.jsonl \
  --output data/processed/embeddings/mert_95
```

为了复现实验，每个 embedding 目录应保存 config_snapshot.yaml，记录模型名、采样率、池化方式、代码版本、生成时间。

## P 主 profile 构建

每位 P 主的 profile 不是一个标签，而是一组原型向量。

最简单 profile 是 mean centroid：

```python
producer_embedding = mean(all_segment_embeddings_of_this_producer)
```

更推荐 KMeans profile：

```python
producer_centroids = KMeans(n_clusters=5).fit(all_segment_embeddings).cluster_centers_
```

检索时，用户歌曲 embedding 与某位 P 主的多个 centroids 分别计算相似度，取最大值或 top-m 平均。这样可以表达“这个 P 主有多个风格面”。

build_profiles.py 应输出：

```python
{
  "backend": "mert_95",
  "producers": {
    "wowaka": {
      "display_name": "wowaka",
      "centroids": np.ndarray,
      "song_count": 30,
      "segment_count": 240
    }
  }
}
```

## 相似度计算

推荐使用余弦相似度。所有 embedding 在保存前或检索前都做 L2 normalize。

用户上传歌曲会得到多个 segment embeddings。可以有三种聚合方式。

第一种是 mean pooling：所有片段平均，然后和 P 主 centroid 比较。简单稳定，但容易被前奏、尾奏、低信息段稀释。

第二种是 segment-level top-k：每个片段分别和 P 主 centroids 比较，然后对最高的一部分相似度取平均。它更符合“只要某些核心段落很像，就给较高分”的娱乐体验。

第三种是 attention pooling：训练一个小模型学习哪些片段更重要。第一版不建议上来做。

MVP 推荐使用 segment-level top-k。伪代码：

```python
def score_song_against_producer(song_segment_embs, producer_centroids, top_ratio=0.4):
    sims = cosine_similarity(song_segment_embs, producer_centroids)
    per_segment_best = sims.max(axis=1)
    k = max(1, int(len(per_segment_best) * top_ratio))
    top_scores = sorted(per_segment_best, reverse=True)[:k]
    return float(np.mean(top_scores))
```

最终对所有 P 主计算分数，排序输出 Top-K。

## 训练版 probe

当无训练版跑通后，可以训练一个轻量 probe。输入是片段 embedding，标签是 producer_slug。模型可以从 logistic regression 开始，再尝试 MLP。

训练集切分必须按 song_id，而不是按 segment_id。否则同一首歌的不同片段可能同时出现在训练和测试中，测试准确率会虚高。

split.py 应生成：

```text
train_song_ids.txt
val_song_ids.txt
test_song_ids.txt
```

训练命令示例：

```bash
python scripts/train_probe.py \
  --embedding_dir data/processed/embeddings/mert_95 \
  --split_dir data/processed/splits \
  --model mlp \
  --output runs/probe_mert95_v1
```

MLP 结构可以非常简单：

```python
Linear(embedding_dim, 512)
ReLU
Dropout(0.2)
Linear(512, num_producers)
```

评价时重点看 Top-1、Top-3、Top-5 accuracy，以及 producer-level confusion matrix。不要只看 overall accuracy，因为数据量不平衡时它会误导。

如果分类器准确率明显高于 centroid baseline，说明 embedding 中确实存在可学习的 P 主差异。如果分类器训练准确率很高但验证很差，说明数据量不足或泄漏/捷径严重。

## 评价协议

至少做三种评价。

第一种是歌曲级随机切分，但必须保证同一首歌的所有片段只出现在一个 split 中。

第二种是年份切分。例如用某位 P 主早期作品作为参考库，测试后期作品。这能检查模型是否只学到了某个时期的音色。

第三种是人工听感评价。随机抽取用户曲或测试曲，让人判断 Top-5 是否“有道理”。因为项目是娱乐向，人工主观评价非常重要。

推荐指标包括 Top-1 accuracy、Top-3 accuracy、Top-5 accuracy、mean reciprocal rank、confusion matrix、per-producer accuracy、平均相似度分布、错误案例分析。

错误案例要记录到 docs/error_analysis.md，例如：

```text
Case 001:
输入歌曲：...
真实 P 主：Neru
模型 Top-5：kemu, Neru, wowaka, かいりきベア, DECO*27
分析：高能摇滚和副歌爆发导致 kemu 分数更高，这个错误在听感上可接受。

Case 002:
输入歌曲：...
真实 P 主：Orangestar
模型 Top-5：MIMI, 40mP, Orangestar, ...
分析：钢琴和透明感特征占主导，但节奏推进没有被模型充分利用。
```

## API 设计

后端推荐使用 FastAPI。第一版 API 只需要四类接口：上传音频、查询任务状态、返回相似结果、查询 P 主信息。

接口设计如下：

```text
POST /api/analyze
上传音频文件，返回 job_id 或直接返回结果。

GET /api/jobs/{job_id}
查询分析状态。

GET /api/producers
返回当前参考库中的 P 主列表。

GET /api/producers/{producer_slug}
返回某位 P 主的基本信息和参考曲数量。
```

如果第一版处理速度足够快，可以 POST /api/analyze 同步返回结果。如果上传歌曲较长或 CPU 推理较慢，应使用异步任务队列。简单版可以用后台线程或 asyncio，正式版可以用 Celery/RQ + Redis。

请求示例：

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@example.wav"
```

返回示例：

```json
{
  "job_id": "job_abc123",
  "status": "done",
  "result": {
    "top_k": [
      {
        "producer_slug": "kemu",
        "display_name": "kemu",
        "score": 0.84,
        "rank": 1
      },
      {
        "producer_slug": "neru",
        "display_name": "Neru",
        "score": 0.79,
        "rank": 2
      }
    ],
    "warnings": []
  }
}
```

FastAPI 主入口 src/vocaptest/api/main.py：

```python
from fastapi import FastAPI, UploadFile, File
from vocaptest.api.schemas import AnalyzeResponse
from vocaptest.pipeline import analyze_uploaded_file

app = FastAPI(title="Vocaloid Producer Style API")

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    result = await analyze_uploaded_file(file)
    return result

@app.get("/api/producers")
def list_producers():
    return {"producers": []}
```

第一版可以把 pipeline 写成同步函数，后续再改成任务队列。

## 前端设计

前端推荐使用 Next.js 或 Vite + React。第一版页面很简单：上传音频、显示分析进度、展示 Top-5 P 主及相似度分数。

页面结构可以是：

 + "`	ext" + @"
/
  首页，项目说明，上传入口。

/result/{job_id}
  分析结果页，显示 Top-5 P 主和分数。

/producers
  当前参考库 P 主列表。
 + "`" + @"

结果页展示：

"你的曲风最接近：kemu（相似度 0.84）"
"同时也接近：Neru（0.79）、wowaka（0.75）、かいりきベア（0.72）、DECO*27（0.68）"
"注意：结果基于当前参考曲库，不代表严格音乐学判断。"

## 传统 MIR 特征（可选调试工具）

如果后续需要调试和内部验证，可以提取一些传统 MIR 特征。这些特征不用于前端展示，仅用于内部检查模型是否被音色等非风格因素带偏。可以使用 librosa 提取：BPM、onset density、spectral centroid、spectral bandwidth、zero crossing rate、RMS loudness、MFCC mean/std 等。第一版不强制实现。

## 可参考或克隆的仓库

建议克隆三个仓库到 external/，但第一版不要直接依赖它们的内部代码，主要用于参考模型调用、训练范式和实验组织。

第一个是 MERT 官方仓库。用途是了解 MERT 模型背景、可用权重、MARBLE benchmark 评估方式。第一版不需要训练 MERT，也不需要 fairseq 训练代码。只需要通过 Hugging Face 加载 MERT-v1-95M 或 MERT-v1-330M。

```bash
git clone https://github.com/yizhilll/MERT external/MERT
```

第二个是 MuQ 官方仓库。用途是参考 MuQ 和 MuQ-MuLan 的加载方式、24kHz 输入要求、音乐文本联合 embedding。MuQ 可作为第二个音频 embedding baseline，MuQ-MuLan 可作为风格文本解释辅助模型。

```bash
git clone https://github.com/tencent-ailab/MuQ external/MuQ
```

第三个是 FM-music-tagging。这个仓库对你的工程最有参考价值，因为它已经把 MERT、CLAP-Music、Qwen2-Audio 作为 foundation backbone，并提供 MLP probe、supervised fine-tuning、few-shot evaluation 的组织方式。你可以参考它的 backbone wrapper、训练脚本结构和评估方式，把数据集从 music tagging 改成 producer classification 或 producer retrieval。

```bash
git clone https://github.com/pxaris/FM-music-tagging external/FM-music-tagging
```

可选参考仓库包括 music-artist-classification-crnn，用来参考传统 artist classification 中的片段预测与歌曲级 majority vote 思路。但你的第一版不建议从 CRNN 开始，因为预训练 embedding 路线更省数据、更快。

## 开发里程碑

第一阶段目标是跑通离线检索 demo。具体任务是：确定 10 位 P 主，手动准备每人 10 首歌，统一转 wav，切片，用 MERT-v1-95M 提 embedding，构建每位 P 主 mean centroid，写一个命令行脚本输入本地 wav，输出 Top-5。这个阶段不需要前端，不需要 API，不需要训练。

验收标准是：输入某位 P 主的未进入参考库的歌曲，Top-5 中能出现本人或听感接近者；输入风格明显的歌曲时，输出结果在主观上不离谱。

第二阶段目标是建立标准数据管线。具体任务是：接入 VocaDB 元数据，建立 song_links.jsonl，加入数据清洗状态，写音频 manifest，加入 embedding 缓存，加入 KMeans 多 centroid profile。这个阶段要避免手工散乱文件。

验收标准是：删除 processed 目录后，可以从 raw metadata 和 raw audio 自动重建 segments、embeddings、profiles。

第三阶段目标是建立 API 和简单前端。具体任务是：FastAPI 上传音频，后端调用同一套 pipeline，返回 Top-5 JSON。前端做上传页面和结果页。

验收标准是：用户上传一首 wav/mp3，页面能显示 Top-5 P 主和分数。

第四阶段目标是加入评估。具体任务是：按 song_id 切 train/val/test，计算 centroid baseline 的 Top-K accuracy，绘制 confusion matrix，用 UMAP 可视化 P 主 embedding 分布。

验收标准是：docs/evaluation_protocol.md 中记录数据版本、模型版本、切分方式、指标结果和错误案例。

第五阶段目标是加入轻量训练。具体任务是：固定 MERT/MuQ embedding，训练 logistic regression、SVM 或 MLP probe，与 centroid baseline 比较。只有当训练版泛化更好时，才把它接入产品。

验收标准是：验证集 Top-3 或 Top-5 明显优于 centroid baseline，且错误案例没有明显数据泄漏。

第六阶段目标是对外部署上线。具体任务是：部署到云服务器，配置域名和 HTTPS，添加速率限制，添加使用统计。

验收标准是：外部用户可以通过公网 URL 访问和使用服务。

## 最小命令流

初始化目录：

```bash
python scripts/00_init_dirs.py
```

拉取或整理元数据：

```bash
python scripts/01_fetch_vocadb_metadata.py \
  --producers configs/producers.yaml \
  --output data/raw/metadata/song_links.jsonl
```

准备音频：

```bash
python scripts/02_prepare_audio.py \
  --metadata data/raw/metadata/song_links.jsonl \
  --output data/interim
```

提取 embedding：

```bash
python scripts/03_extract_embeddings.py \
  --config configs/model_mert.yaml \
  --segments data/interim/segments/segments.jsonl \
  --output data/processed/embeddings/mert_95
```

构建 P 主 profiles：

```bash
python scripts/04_build_profiles.py \
  --embeddings data/processed/embeddings/mert_95 \
  --clusters 5 \
  --output data/processed/producer_profiles/profiles_mert_95.pkl
```

命令行测试：

```bash
python scripts/05_evaluate_retrieval.py \
  --profile data/processed/producer_profiles/profiles_mert_95.pkl \
  --input examples/user_song.wav \
  --top_k 5
```

启动 API：

```bash
uvicorn vocaptest.api.main:app --host 0.0.0.0 --port 8000
```

## 第一版代码模块职责

vocadb_client.py 负责请求 VocaDB API，搜索 artist，拉取 song metadata，保存原始 JSON。它不负责下载音频，也不负责模型处理。

build_song_index.py 负责把 VocaDB 原始结果转成项目内部统一 metadata schema，并标记 accepted/rejected/pending。

download_audio.py 负责根据 song_links.jsonl 下载音频或登记本地音频文件路径。

preprocess.py 负责音频格式统一。

segment.py 负责切片和静音过滤。

mert_embedder.py、muq_embedder.py、clap_embedder.py 负责模型加载和 embedding 提取。

extract_embeddings.py 负责批量提取并缓存 embedding。

build_profiles.py 负责把片段级 embedding 聚合为 P 主 profile。

similarity.py 负责余弦相似度、top-k 聚合、分数校准。

search.py 负责输入一首歌，返回 Top-K P 主。

api/main.py 负责 FastAPI 服务入口。

routes_upload.py 负责上传与分析接口。

routes_metadata.py 负责 P 主列表接口。

## 推荐的数据 schema

Producer：

```python
class Producer:
    slug: str
    display_name: str
    vocadb_artist_id: int | None
    aliases: list[str]
    notes: str | None
```

Song：

```python
class Song:
    song_id: str
    producer_slug: str
    title: str
    publish_date: str | None
    source_urls: list[str]
    vocalists: list[str]
    tags: list[str]
    is_cover: bool
    is_remix: bool
    is_collaboration: bool
    status: str
    local_audio_path: str | None
```

Segment：

```python
class Segment:
    segment_id: str
    song_id: str
    producer_slug: str
    path: str
    start_sec: float
    end_sec: float
    duration_sec: float
    rms_db: float
```

EmbeddingRecord：

```python
class EmbeddingRecord:
    segment_id: str
    song_id: str
    producer_slug: str
    model_backend: str
    embedding_path: str
    embedding_dim: int
```

SearchResult：

```python
class SearchResult:
    producer_slug: str
    display_name: str
    score: float
    rank: int
```

## .env 准备

第一版不一定需要复杂 API key，但建议准备 .env.example：

```env
VOCADB_BASE_URL=https://vocadb.net/api
VOCADB_USER_AGENT=vocaloid-producer-style-demo/0.1 contact:your_email@example.com

HF_HOME=.cache/huggingface
TRANSFORMERS_CACHE=.cache/huggingface

API_HOST=0.0.0.0
API_PORT=8000

MAX_UPLOAD_MB=50
```

如果后续使用 YouTube Data API、Niconico API 或对象存储，再加入对应 key。第一版如果只用 yt-dlp，不需要 YouTube API key。

## 重要工程原则

第一，所有中间结果都要可缓存。音频下载、重采样、切片、embedding、profile 都应该可复用。

第二，所有数据都要有 manifest。不要只靠文件夹里的文件名猜测标签。标签必须来自 metadata jsonl。

第三，所有实验都要保存 config snapshot。否则你无法知道某个结果是 MERT 还是 MuQ，是 20 秒切片还是 30 秒切片，是 mean pooling 还是 top-k pooling。

第四，不要在第一版微调大模型。先证明 frozen embedding + centroid 是否有效。如果这个 baseline 都没有主观效果，微调大模型很可能只是把数据集噪声学得更牢。

第五，不要用 segment 随机切分做评估。必须按 song_id 切分。

第六，结果表述要避免绝对化。页面应该说“在当前参考库中最接近”，不要说“你的曲风就是某某 P 主”。

## 后续扩展方向

后续可以加入人声分离，把 full mix、vocal stem、instrumental stem 分别提 embedding。这样可以区分“整体像谁”“伴奏像谁”“调声像谁”。对于 Vocaloid 场景，这会非常有价值，因为 P 主风格常常同时体现在作曲、编曲、混音和调声上。

也可以加入歌词 embedding。很多 P 主的风格不仅来自音频，还来自歌词主题、叙事方式和词汇习惯。可以把歌词通过 multilingual text embedding 模型编码，与音频结果融合。但歌词获取和清洗会增加复杂度，建议放到第二阶段之后。

还可以加入用户反馈。结果页可以让用户选择“准”“有点像”“不像”，并记录匿名反馈。长期可以用这些反馈校准相似度分数，甚至训练个性化偏好模型。

最终版本可以从“P 主分类器”变成“Vocaloid 风格地图”。P 主只是地图上的若干代表点，用户上传的歌曲落在某个区域中。这样比硬分类更符合音乐风格的连续性，也更适合娱乐产品。
