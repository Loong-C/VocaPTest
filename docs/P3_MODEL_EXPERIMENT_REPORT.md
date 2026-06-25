# P3 模型实验报告

日期：2026-06-25

## 目标

本实验只测试未来继续增加 P 主时也应该有效的全局模型头改法；不针对某两个作者写特殊规则，也不修改部署链路或 VPS 文件。

## 有效修正

真正有效的改法很朴素：保留已经缓存好的 MERT-95M，不微调 backbone；在若干个 MERT 层上分别训练歌曲级 Shrinkage LDA，然后平均这些层的类别概率。按 dev macro-F1 选择出的三层是 **5、6、8**：第 6 层是当前主线 baseline，第 8 层是 dev 上最强的单层，第 5 层是 dev macro-F1 次优单层。这是一条全局选择规则，不是给某个 P 主写补丁。

## 效果

| 方法 | Dev Top-1 | Final Top-1 | Final Top-3 | Final Macro-F1 | Final 高置信错误 |
|---|---:|---:|---:|---:|---:|
| 当前第 6 层 LDA baseline | 75.81% | 78.23% | 92.74% | 78.24% | 3 |
| 推荐的 5/6/8 层集成 | 77.42% | 82.26% | 93.55% | 82.20% | 3 |

相对当前单层 baseline，推荐方案在 final frozen test 上的变化是：

- Top-1：+4.03 pp；错误曲目从 27 首降到 22 首。
- Macro-F1：+3.96 pp；提升不是单个大类带来的偶然收益。
- Top-3：+0.81 pp；没有牺牲产品最重要的候选列表体验。
- 拒识：把 OOF 目标接受精度从 0.95 提到 0.96 后，final 高置信错误维持在 3 首，被接受样本准确率为 96.15%。

## 失败或不推荐的方向

- 更宽的中层窗口 [3, 4, 5, 6, 7, 8, 9] 虽然给出最高 dev Top-1 (79.03%)，但 final Top-3 降到 91.13%。它把层间信息平均得过头，排序变钝。
- mean+std 池化看起来能描述歌曲内部变化，但最佳 probe 只有 dev Top-1 75.81%。在当前数据规模下，它增加噪声维度的速度快于增加稳定风格信息的速度。
- 双原型 probe 在探索运行中明显低于 baseline。当前每类十几首歌不足以稳定拆出子簇，除非未来每类歌曲显著增多，或先训练出更好的 metric-learning 投影，否则不应作为主线。

## 落地状态

5/6/8 层集成已经补齐正式训练脚本，并保存为 API 可直接加载的 `LayerFusionLDA` artifact：

- 模型：`data/processed/models/p1_layer_fusion_lda.pkl`
- 配置：`configs/retrieval.yaml` 的 `p1_model_path`
- 生产训练脚本：`scripts/21_train_p3_layer_fusion.py`
- 部署评估产物：`data/processed/evaluations/p3_layer_fusion_deploy.json`

需要注意：此前仓库里的 `p1_layer_fusion_lda.pkl` 是更早阶段留下的 20 类旧模型，不能代表本报告里的 31 类 P3 方案；上线前必须重新运行生产训练脚本并同步新的模型文件。

## 复现

```powershell
$env:PYTHONPATH='src'
python scripts/20_run_p3_model_experiments.py
python scripts/21_train_p3_layer_fusion.py
```

产物：

- `data/processed/evaluations/p3_model_experiments.json`
- `data/processed/evaluations/p3_layer_fusion_deploy.json`
- `docs/P3_MODEL_EXPERIMENT_REPORT.md`
