# P4 Stacking Search

Protocol: base heads create train OOF probabilities; meta heads are selected on dev. Final is report-only.

## Selected Candidate

| Field | Value |
|---|---|
| meta | `{"alpha": 30.0, "feature_mode": "prob_and_log", "kind": "ridge"}` |
| Dev Top-1 | 82.98% |
| Dev Macro-F1 | 78.67% |
| Final Top-1 | 80.53% |
| Final Top-3 | 88.42% |
| Final Macro-F1 | 77.41% |
| Target met | `False` |

## Success Gates

| Gate | Passed |
|---|---:|
| `top1_plus_4pp_guarded` | `False` |
| `top3_plus_3pp_guarded` | `False` |
| `macro_f1_plus_4pp_guarded` | `False` |
| `balanced_plus_3_1p5_3pp` | `False` |

## Top Meta Candidates

| Rank | Candidate | Dev Top-1 | Dev Macro-F1 | Final Top-1 | Final Top-3 | Final Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `{"alpha": 30.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | 82.98% | 78.67% | 80.53% | 88.42% | 77.41% |
| 2 | `{"alpha": 10.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | 82.98% | 78.67% | 80.53% | 86.84% | 79.01% |
| 3 | `{"alpha": 30.0, "feature_mode": "log_prob", "kind": "ridge"}` | 82.98% | 78.60% | 79.47% | 88.42% | 76.02% |
| 4 | `{"C": 0.3, "feature_mode": "prob", "kind": "logreg"}` | 81.91% | 78.07% | 80.53% | 83.16% | 78.02% |
| 5 | `{"alpha": 30.0, "feature_mode": "prob", "kind": "ridge"}` | 81.91% | 78.00% | 80.00% | 84.74% | 77.38% |
| 6 | `{"alpha": 10.0, "feature_mode": "prob", "kind": "ridge"}` | 81.91% | 78.00% | 80.53% | 84.74% | 77.63% |
| 7 | `{"alpha": 3.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | 81.91% | 77.81% | 78.42% | 84.21% | 76.07% |
| 8 | `{"alpha": 3.0, "feature_mode": "log_prob", "kind": "ridge"}` | 81.91% | 77.67% | 78.42% | 87.37% | 74.95% |
| 9 | `{"alpha": 10.0, "feature_mode": "log_prob", "kind": "ridge"}` | 81.91% | 77.53% | 79.47% | 87.89% | 76.04% |
| 10 | `{"C": 0.03, "feature_mode": "prob", "kind": "logreg"}` | 81.91% | 77.33% | 80.53% | 83.16% | 77.92% |
| 11 | `{"C": 0.03, "feature_mode": "prob_and_log", "kind": "logreg"}` | 81.91% | 77.00% | 81.58% | 87.89% | 79.97% |
| 12 | `{"C": 0.1, "feature_mode": "prob_and_log", "kind": "logreg"}` | 81.91% | 76.81% | 80.53% | 87.37% | 78.69% |
| 13 | `{"C": 0.3, "feature_mode": "prob_and_log", "kind": "logreg"}` | 81.91% | 76.81% | 81.05% | 88.42% | 79.00% |
| 14 | `{"C": 1.0, "feature_mode": "prob", "kind": "logreg"}` | 80.85% | 76.73% | 79.47% | 84.21% | 77.11% |
| 15 | `{"C": 1.0, "feature_mode": "prob_and_log", "kind": "logreg"}` | 81.91% | 76.67% | 80.53% | 88.42% | 78.47% |

## Base Heads

| Base | OOF Top-1 | Dev Top-1 | Final Top-1 | Final Top-3 | Final Macro-F1 |
|---|---:|---:|---:|---:|---:|
| `layer_6` | 82.37% | 84.04% | 77.37% | 87.89% | 74.32% |
| `layer_7` | 82.37% | 78.72% | 77.89% | 88.95% | 73.66% |
| `layer_8` | 81.50% | 77.66% | 77.89% | 88.95% | 75.02% |
| `fusion_567` | 82.90% | 82.98% | 79.47% | 89.47% | 76.31% |
| `fusion_568` | 83.60% | 82.98% | 78.42% | 88.42% | 75.55% |
| `concat_56` | 83.94% | 81.91% | 80.00% | 87.89% | 76.58% |
| `concat_67` | 84.29% | 80.85% | 81.58% | 88.95% | 78.88% |
| `concat_567` | 85.34% | 81.91% | 80.53% | 89.47% | 77.47% |
| `concat_568` | 85.17% | 80.85% | 80.00% | 88.95% | 76.89% |
| `concat_678` | 85.17% | 79.79% | 80.53% | 89.47% | 78.02% |
| `concat_789` | 83.25% | 80.85% | 80.00% | 87.89% | 77.62% |
| `concat_45678` | 85.34% | 80.85% | 81.58% | 88.95% | 78.79% |

## Reproduce

```powershell
python scripts/31_run_p4_stacking_search.py
```
