# P4 Candidate Cross-Validation

这里只在训练集内部做 grouped cross-validation，用来判断 broad search 里几个候选方向是否只是 dev/final 偶然波动。

| 过滤 | 候选 | CV Top-1 mean | CV Top-1 std | CV Top-3 mean | CV Macro-F1 mean | CV LogLoss mean |
|---|---|---:|---:|---:|---:|---:|
| raw | baseline_lda_568 | 82.69% | 0.61% | 91.70% | 83.37% | 2.222 |
| raw | raw_concat_lda_mid_4_8 | 84.84% | 0.60% | 91.06% | 85.43% | 3.155 |
| source_clean | source_clean_lda_top3_567 | 82.32% | 0.61% | 90.75% | 82.76% | 2.308 |
| source_clean | source_clean_temperature_1567 | 82.37% | 0.87% | 90.23% | 82.76% | 1.141 |
| source_clean | source_clean_concat_lda_567 | 84.41% | 0.96% | 91.04% | 85.06% | 3.146 |
| review_clean | review_clean_lda_568 | 81.94% | 1.09% | 90.73% | 81.77% | 2.489 |
| review_clean | review_clean_temperature_236811 | 80.15% | 0.84% | 89.19% | 80.21% | 0.979 |

## 复现

```powershell
python scripts/28_validate_p4_broad_candidates.py
```
