# P 主风格标签来源

前端展示的 P 主风格标签已从主观硬编码迁移到
`configs/producer_style_tags.yaml`。当前口径是 **VocaDB song tags**：
每位 P 主关联其 VocaDB artist 页面，并缓存 3 个用于展示的风格标签。

这些标签只用于页面展示、搜索提示和结果解释，不参与音频模型训练、模型选择或评估。
模型仍只依赖音频分段的 MERT 表征与 LDA 分类头。

## 为什么要缓存

VocaDB 目前可能对非浏览器请求触发 Cloudflare challenge。为了避免部署后页面
依赖实时第三方请求，生产运行时不会访问 VocaDB。后端只读取仓库内的
`configs/producer_style_tags.yaml`，因此 VPS 部署和分析接口都保持稳定。

## 刷新流程

在能够访问 VocaDB API 的环境，或准备好浏览器导出的
`data/raw_jsonl/<producer_slug>_songs.jsonl` 后，可运行：

```powershell
python scripts/22_refresh_vocadb_style_tags.py `
  --producers configs/producers.yaml `
  --raw-dir data/raw_jsonl `
  --output configs/producer_style_tags.yaml
```

如果当前环境无法访问 VocaDB，脚本会保留已有标签，避免把 UI 标签清空。
脚本会过滤 Vocaloid 声库名、平台名、活动名、PV 类型等非风格标签，并校验
raw JSONL 中的 artist id，防止错误缓存污染结果。

## 维护原则

- 每个 P 主至少保留 3 个展示标签。
- 标签应来自 VocaDB song tag 体系或刷新脚本聚合结果。
- 不把标签当作模型真值，也不把它用于分类器训练。
- 如果引入新 P 主，应同时添加 `configs/producers.yaml` 和
  `configs/producer_style_tags.yaml` 条目。
