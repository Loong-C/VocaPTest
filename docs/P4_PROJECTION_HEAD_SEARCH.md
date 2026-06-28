# P4 Projection Head Search

Protocol: frozen MERT song features, source-clean train set, small regularized projection heads selected on dev. Final is evaluated only for the selected dev candidate.

## Selected Candidate

| Field | Value |
|---|---|
| candidate | `{"contrastive_weight": 0.05, "layers": [6], "learning_rate": 0.001, "projection_dim": 64, "weight_decay": 0.003}` |
| Dev Top-1 | 84.04% |
| Dev Top-3 | 87.23% |
| Dev Macro-F1 | 79.67% |
| Final Top-1 | 71.58% |
| Final Top-3 | 86.84% |
| Final Macro-F1 | 67.88% |
| Target met | `False` |

## Success Gates

| Gate | Passed |
|---|---:|
| `top1_plus_4pp_guarded` | `False` |
| `top3_plus_3pp_guarded` | `False` |
| `macro_f1_plus_4pp_guarded` | `False` |
| `balanced_plus_3_1p5_3pp` | `False` |

## Top Dev Candidates

| Rank | Candidate | Dev Top-1 | Dev Top-3 | Dev Macro-F1 | Mean Epochs |
|---:|---|---:|---:|---:|---:|
| 1 | `{"contrastive_weight": 0.05, "layers": [6], "learning_rate": 0.001, "projection_dim": 64, "weight_decay": 0.003}` | 84.04% | 87.23% | 79.67% | 35.5 |
| 2 | `{"contrastive_weight": 0.05, "layers": [4, 5, 6, 7, 8], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 82.98% | 90.43% | 79.00% | 44.5 |
| 3 | `{"contrastive_weight": 0.0, "layers": [6, 7, 8], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 81.91% | 90.43% | 78.40% | 29.0 |
| 4 | `{"contrastive_weight": 0.0, "layers": [7], "learning_rate": 0.001, "projection_dim": 64, "weight_decay": 0.003}` | 81.91% | 88.30% | 77.67% | 30.0 |
| 5 | `{"contrastive_weight": 0.05, "layers": [6, 7, 8], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 81.91% | 89.36% | 77.53% | 30.0 |
| 6 | `{"contrastive_weight": 0.0, "layers": [6, 7, 8], "learning_rate": 0.001, "projection_dim": 64, "weight_decay": 0.003}` | 81.91% | 89.36% | 76.87% | 29.5 |
| 7 | `{"contrastive_weight": 0.0, "layers": [6], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 80.85% | 90.43% | 76.73% | 21.0 |
| 8 | `{"contrastive_weight": 0.0, "layers": [4, 5, 6, 7, 8], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 81.91% | 88.30% | 76.53% | 36.5 |
| 9 | `{"contrastive_weight": 0.0, "layers": [7], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 80.85% | 85.11% | 76.47% | 43.5 |
| 10 | `{"contrastive_weight": 0.05, "layers": [5, 6, 7], "learning_rate": 0.001, "projection_dim": 64, "weight_decay": 0.003}` | 80.85% | 90.43% | 76.00% | 25.5 |
| 11 | `{"contrastive_weight": 0.05, "layers": [7], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 79.79% | 85.11% | 75.40% | 29.0 |
| 12 | `{"contrastive_weight": 0.0, "layers": [5, 6, 7], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 80.85% | 88.30% | 75.27% | 30.0 |
| 13 | `{"contrastive_weight": 0.1, "layers": [7], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 80.85% | 88.30% | 75.20% | 52.0 |
| 14 | `{"contrastive_weight": 0.05, "layers": [6], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 78.72% | 87.23% | 74.73% | 19.5 |
| 15 | `{"contrastive_weight": 0.05, "layers": [5, 6, 7], "learning_rate": 0.001, "projection_dim": 128, "weight_decay": 0.003}` | 78.72% | 86.17% | 74.48% | 25.0 |

## Reproduce

```powershell
python scripts/32_run_p4_projection_head_search.py
```
