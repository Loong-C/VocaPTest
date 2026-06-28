# P4 CV-Selected Stacking

Protocol: base heads create train OOF probabilities. Meta candidates are selected by train-only grouped CV, with a dev guard. Final is evaluated only for the selected candidate.

## Selected Candidate

| Field | Value |
|---|---|
| meta | `{"alpha": 3.0, "feature_mode": "log_prob", "kind": "ridge"}` |
| CV Top-1 | 85.34% |
| CV Top-3 | 90.75% |
| CV Macro-F1 | 85.55% |
| Dev Top-1 | 81.91% |
| Dev Top-3 | 88.30% |
| Dev Macro-F1 | 77.67% |
| Final Top-1 | 78.42% |
| Final Top-3 | 87.37% |
| Final Macro-F1 | 74.95% |
| Target met | `False` |

## Success Gates

| Gate | Passed |
|---|---:|
| `top1_plus_4pp_guarded` | `False` |
| `top3_plus_3pp_guarded` | `False` |
| `macro_f1_plus_4pp_guarded` | `False` |
| `balanced_plus_3_1p5_3pp` | `False` |

## Top CV Candidates

| Rank | Candidate | Guard | CV Top-1 | CV Top-3 | CV Macro-F1 | Dev Top-1 | Dev Top-3 | Dev Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `{"alpha": 3.0, "feature_mode": "log_prob", "kind": "ridge"}` | `True` | 85.34% | 90.75% | 85.55% | 81.91% | 88.30% | 77.67% |
| 2 | `{"alpha": 30.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 84.99% | 89.88% | 85.35% | 82.98% | 86.17% | 78.67% |
| 3 | `{"alpha": 10.0, "feature_mode": "log_prob", "kind": "ridge"}` | `True` | 84.99% | 91.27% | 85.24% | 81.91% | 89.36% | 77.53% |
| 4 | `{"alpha": 30.0, "feature_mode": "prob", "kind": "ridge"}` | `True` | 84.82% | 88.13% | 85.23% | 81.91% | 84.04% | 78.00% |
| 5 | `{"alpha": 3.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 84.47% | 88.66% | 84.77% | 81.91% | 85.11% | 77.81% |
| 6 | `{"C": 0.1, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 84.64% | 89.53% | 84.71% | 80.85% | 85.11% | 75.81% |
| 7 | `{"alpha": 30.0, "feature_mode": "log_prob", "kind": "ridge"}` | `True` | 84.47% | 91.27% | 84.69% | 82.98% | 90.43% | 78.60% |
| 8 | `{"C": 0.3, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 84.47% | 89.88% | 84.59% | 80.85% | 85.11% | 75.81% |
| 9 | `{"alpha": 10.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 84.29% | 89.35% | 84.56% | 82.98% | 85.11% | 78.67% |
| 10 | `{"C": 0.03, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 84.29% | 90.05% | 84.45% | 80.85% | 85.11% | 75.80% |
| 11 | `{"alpha": 1.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 84.12% | 88.48% | 84.42% | 80.85% | 85.11% | 76.03% |
| 12 | `{"C": 1.0, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 84.12% | 90.05% | 84.28% | 80.85% | 85.11% | 75.81% |
| 13 | `{"C": 3.0, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 84.12% | 90.40% | 84.23% | 80.85% | 85.11% | 75.81% |
| 14 | `{"C": 0.1, "feature_mode": "prob_and_log", "kind": "logreg"}` | `True` | 83.94% | 89.01% | 84.17% | 81.91% | 84.04% | 76.81% |
| 15 | `{"alpha": 0.3, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 83.07% | 87.78% | 83.94% | 80.85% | 84.04% | 75.80% |

## Reproduce

```powershell
python scripts/33_run_p4_cv_selected_stacking.py
```
