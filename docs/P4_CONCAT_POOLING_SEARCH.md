# P4 Concat Pooling Search

Protocol: no new data, no producer-specific rules, no final-based tuning. Candidates are searched on dev and stabilized with train-only grouped CV; final is used only after selection.

Success target: any one guarded gate may pass, but no single metric is allowed to improve while the others collapse.

- Top-1 gate: Top-1 >= 82.42%, Top-3 >= 88.45%, Macro-F1 >= 77.55%.
- Top-3 gate: Top-3 >= 91.95%, Top-1 >= 80.42%, Macro-F1 >= 77.55%.
- Macro gate: Macro-F1 >= 79.55%, Top-1 >= 80.42%, Top-3 >= 88.45%.
- Balanced gate: Top-1 >= 81.42%, Top-3 >= 90.45%, Macro-F1 >= 78.55%.

## Selected Candidate

| Field | Value |
|---|---|
| name | `concat_lda` |
| mode | `mean` |
| kind | `lda` |
| layers | `[4, 5, 6, 7, 8]` |
| pca_dim | `None` |
| standardize | `False` |
| CV Top-1 | 84.82% |
| CV Macro-F1 | 85.38% |
| Dev Top-1 | 80.85% |
| Dev Macro-F1 | 76.61% |
| Final Top-1 | 81.58% |
| Final Top-3 | 88.95% |
| Final Macro-F1 | 78.79% |
| Target met | `False` |

## Success Gates

| Gate | Passed |
|---|---:|
| `top1_plus_4pp_guarded` | `False` |
| `top3_plus_3pp_guarded` | `False` |
| `macro_f1_plus_4pp_guarded` | `False` |
| `balanced_plus_3_1p5_3pp` | `False` |

## Top CV-Checked Candidates

| Rank | Name | Mode | Kind | Layers | PCA | Std | CV Top-1 | CV Macro-F1 | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `concat_lda` | `mean` | `lda` | `[6]` | None | `False` | 81.97% | 82.45% | 84.04% | 80.20% | 77.37% | 87.89% |
| 2 | `concat_lda` | `mean` | `lda` | `[5, 6]` | None | `False` | 82.66% | 83.16% | 81.91% | 78.80% | 80.00% | 87.89% |
| 3 | `concat_lda` | `mean` | `lda` | `[4, 5, 6]` | None | `False` | 82.43% | 83.01% | 81.91% | 78.13% | 78.42% | 88.42% |
| 4 | `concat_lda` | `mean` | `lda` | `[5, 6, 7]` | None | `False` | 84.41% | 85.06% | 81.91% | 77.73% | 80.53% | 89.47% |
| 5 | `concat_lda` | `mean` | `lda` | `[6, 7]` | None | `False` | 84.00% | 84.54% | 80.85% | 76.93% | 81.58% | 88.95% |
| 6 | `concat_ridge` | `mean` | `ridge` | `[1, 5, 6, 7]` | None | `True` | 82.49% | 82.37% | 80.85% | 76.80% | 78.42% | 84.74% |
| 7 | `concat_ridge` | `mean` | `ridge` | `[1, 5, 6, 7]` | None | `True` | 81.33% | 81.10% | 80.85% | 76.80% | 77.37% | 84.21% |
| 8 | `concat_lda` | `mean` | `lda` | `[7, 8, 9]` | None | `False` | 83.71% | 84.68% | 80.85% | 76.68% | 80.00% | 87.89% |
| 9 | `concat_lda` | `mean` | `lda` | `[4, 5, 6, 7, 8]` | None | `False` | 84.82% | 85.38% | 80.85% | 76.61% | 81.58% | 88.95% |
| 10 | `concat_lda` | `mean` | `lda` | `[5, 6, 8]` | None | `False` | 84.53% | 85.04% | 80.85% | 76.27% | 80.00% | 88.95% |

## Best Dev Candidates Before CV

| Rank | Name | Mode | Kind | Layers | PCA | Std | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|
| 1 | `concat_lda` | `mean` | `lda` | `[6]` | None | `False` | 84.04% | 80.20% | 77.37% | 87.89% |
| 2 | `concat_lda` | `mean` | `lda` | `[5, 6]` | None | `False` | 81.91% | 78.80% | 80.00% | 87.89% |
| 3 | `concat_lda` | `mean` | `lda` | `[4, 5, 6]` | None | `False` | 81.91% | 78.13% | 78.42% | 88.42% |
| 4 | `concat_lda` | `mean` | `lda` | `[5, 6, 7]` | None | `False` | 81.91% | 77.73% | 80.53% | 89.47% |
| 5 | `concat_lda` | `mean` | `lda` | `[6, 7]` | None | `False` | 80.85% | 76.93% | 81.58% | 88.95% |
| 6 | `concat_ridge` | `mean` | `ridge` | `[1, 5, 6, 7]` | None | `True` | 80.85% | 76.80% | 78.42% | 84.74% |
| 7 | `concat_ridge` | `mean` | `ridge` | `[1, 5, 6, 7]` | None | `True` | 80.85% | 76.80% | 77.37% | 84.21% |
| 8 | `concat_lda` | `mean` | `lda` | `[7, 8, 9]` | None | `False` | 80.85% | 76.68% | 80.00% | 87.89% |
| 9 | `concat_lda` | `mean` | `lda` | `[4, 5, 6, 7, 8]` | None | `False` | 80.85% | 76.61% | 81.58% | 88.95% |
| 10 | `concat_lda` | `mean` | `lda` | `[5, 6, 8]` | None | `False` | 80.85% | 76.27% | 80.00% | 88.95% |
| 11 | `concat_lda` | `mean` | `lda` | `[7, 8]` | None | `False` | 79.79% | 75.61% | 81.05% | 88.95% |
| 12 | `concat_lda` | `mean` | `lda` | `[6, 7, 8]` | None | `False` | 79.79% | 75.47% | 80.53% | 89.47% |
| 13 | `concat_lda` | `mean` | `lda` | `[5]` | None | `False` | 79.79% | 75.34% | 74.74% | 87.89% |
| 14 | `concat_pca_lda` | `mean` | `lda` | `[6, 7, 8]` | 128 | `True` | 78.72% | 74.80% | 76.84% | 88.42% |
| 15 | `concat_lda` | `mean` | `lda` | `[7]` | None | `False` | 78.72% | 74.08% | 77.89% | 88.95% |
| 16 | `concat_pca_lda` | `mean` | `lda` | `[5, 6, 7, 8, 9]` | 128 | `True` | 77.66% | 73.60% | 77.37% | 87.89% |
| 17 | `concat_lda` | `mean` | `lda` | `[8]` | None | `False` | 77.66% | 73.53% | 77.89% | 88.95% |
| 18 | `concat_lda` | `mean` | `lda` | `[4, 5]` | None | `False` | 77.66% | 73.48% | 77.37% | 88.42% |
| 19 | `concat_ridge` | `mean` | `ridge` | `[4, 5, 6, 7, 8]` | None | `True` | 77.66% | 73.13% | 78.42% | 85.79% |
| 20 | `concat_ridge` | `mean` | `ridge` | `[5, 6, 8]` | None | `True` | 78.72% | 72.88% | 77.37% | 83.68% |
| 21 | `concat_lda` | `mean` | `lda` | `[8, 9, 10]` | None | `False` | 76.60% | 72.80% | 78.42% | 86.84% |
| 22 | `concat_pca_lda` | `mean` | `lda` | `[4, 5, 6, 7, 8]` | 128 | `True` | 76.60% | 72.48% | 76.32% | 86.84% |
| 23 | `concat_pca_lda` | `mean` | `lda` | `[2, 3, 4, 5, 6]` | 128 | `True` | 76.60% | 72.13% | 71.05% | 85.79% |
| 24 | `concat_lda` | `mean` | `lda` | `[9]` | None | `False` | 75.53% | 72.07% | 75.26% | 85.26% |

## Reproduce

```powershell
python scripts/29_run_p4_concat_pooling_search.py
```
