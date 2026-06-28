# P4 Similarity Search

Protocol: no new data, no producer-specific rules, no final-based tuning. Dev ranks candidates; train-only grouped CV picks the final candidate.

## Selected Candidate

| Field | Value |
|---|---|
| method | `knn` |
| layers | `[6]` |
| transform | `standard_l2` |
| params | `{"k": 15, "layers": [6], "method": "knn", "temperature": 0.1, "transform": "standard_l2", "weighted": true}` |
| CV Top-1 | 64.75% |
| CV Macro-F1 | 63.39% |
| Dev Top-1 | 69.15% |
| Dev Macro-F1 | 64.09% |
| Final Top-1 | 58.95% |
| Final Top-3 | 76.32% |
| Final Macro-F1 | 56.22% |
| Target met | `False` |

## Success Gates

| Gate | Passed |
|---|---:|
| `top1_plus_4pp_guarded` | `False` |
| `top3_plus_3pp_guarded` | `False` |
| `macro_f1_plus_4pp_guarded` | `False` |
| `balanced_plus_3_1p5_3pp` | `False` |

## Top CV-Checked Candidates

| Rank | Method | Layers | Transform | CV Top-1 | CV Macro-F1 | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `knn` | `[6, 7]` | `standard_l2` | 63.35% | 61.91% | 69.15% | 64.58% | 58.95% | 74.21% | 56.30% |
| 2 | `knn` | `[5, 6]` | `standard_l2` | 62.30% | 61.35% | 69.15% | 64.27% | 56.32% | 75.26% | 54.10% |
| 3 | `knn` | `[6]` | `standard_l2` | 64.75% | 63.39% | 69.15% | 64.09% | 58.95% | 76.32% | 56.22% |
| 4 | `knn` | `[6]` | `standard_l2` | 62.13% | 60.92% | 69.15% | 63.65% | 56.32% | 75.79% | 53.83% |

## Reproduce

```powershell
python scripts/30_run_p4_similarity_search.py
```
