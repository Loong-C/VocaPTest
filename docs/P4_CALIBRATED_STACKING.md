# P4 Calibrated Stacking

Protocol: base heads create train OOF probabilities. Meta candidates are selected by train-only grouped CV log-loss/MRR with a dev guard. Final is evaluated only for the selected candidate.

## Selected Candidate

| Field | Value |
|---|---|
| meta | `{"C": 0.03, "feature_mode": "log_prob", "kind": "logreg"}` |
| CV Log Loss | 0.7711 |
| CV MRR | 0.8807 |
| CV Top-3 | 90.05% |
| Dev Top-1 | 80.85% |
| Dev Top-3 | 85.11% |
| Dev Macro-F1 | 75.80% |
| Final Top-1 | 82.11% |
| Final Top-3 | 90.00% |
| Final Macro-F1 | 80.00% |
| Target met | `True` |

## Success Gates

| Gate | Passed |
|---|---:|
| `top1_plus_4pp_guarded` | `False` |
| `top3_plus_3pp_guarded` | `False` |
| `macro_f1_plus_4pp_guarded` | `True` |
| `balanced_plus_3_1p5_3pp` | `False` |

## Top Calibrated Candidates

| Rank | Candidate | Guard | CV Log Loss | CV MRR | CV Top-3 | Dev Top-1 | Dev Top-3 | Dev Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `{"C": 0.03, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 0.7711 | 0.8807 | 90.05% | 80.85% | 85.11% | 75.80% |
| 2 | `{"C": 0.1, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 0.8214 | 0.8812 | 89.53% | 80.85% | 85.11% | 75.81% |
| 3 | `{"C": 0.1, "feature_mode": "prob_and_log", "kind": "logreg"}` | `True` | 0.8919 | 0.8741 | 89.01% | 81.91% | 84.04% | 76.81% |
| 4 | `{"C": 0.3, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 0.9023 | 0.8795 | 89.88% | 80.85% | 85.11% | 75.81% |
| 5 | `{"C": 0.3, "feature_mode": "prob_and_log", "kind": "logreg"}` | `True` | 0.9744 | 0.8711 | 89.01% | 81.91% | 84.04% | 76.81% |
| 6 | `{"C": 1.0, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 1.0132 | 0.8780 | 90.05% | 80.85% | 85.11% | 75.81% |
| 7 | `{"C": 1.0, "feature_mode": "prob_and_log", "kind": "logreg"}` | `True` | 1.0854 | 0.8718 | 88.66% | 81.91% | 84.04% | 76.67% |
| 8 | `{"C": 3.0, "feature_mode": "log_prob", "kind": "logreg"}` | `True` | 1.1313 | 0.8770 | 90.40% | 80.85% | 85.11% | 75.81% |
| 9 | `{"C": 3.0, "feature_mode": "prob_and_log", "kind": "logreg"}` | `True` | 1.2053 | 0.8713 | 88.66% | 81.91% | 84.04% | 76.67% |
| 10 | `{"alpha": 3.0, "feature_mode": "log_prob", "kind": "ridge"}` | `True` | 2.3444 | 0.8854 | 90.75% | 81.91% | 88.30% | 77.67% |
| 11 | `{"alpha": 1.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 2.3550 | 0.8716 | 88.48% | 80.85% | 85.11% | 76.03% |
| 12 | `{"alpha": 3.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 2.3568 | 0.8758 | 88.66% | 81.91% | 85.11% | 77.81% |
| 13 | `{"alpha": 0.3, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 2.3629 | 0.8633 | 87.78% | 80.85% | 84.04% | 75.80% |
| 14 | `{"alpha": 10.0, "feature_mode": "prob_and_log", "kind": "ridge"}` | `True` | 2.3675 | 0.8762 | 89.35% | 82.98% | 85.11% | 78.67% |
| 15 | `{"alpha": 10.0, "feature_mode": "log_prob", "kind": "ridge"}` | `True` | 2.3785 | 0.8853 | 91.27% | 81.91% | 89.36% | 77.53% |

## Reproduce

```powershell
python scripts/34_run_p4_calibrated_stacking.py
```
