# P0 Implementation Report

Date: 2026-06-11

## Delivered

- Auditable song exclusions and work-level canonicalization in
  `configs/dataset_curation.yaml`.
- Portable embedding paths and strict record/vector alignment.
- Training/inference matched segmentation: 24 kHz mono, 20 second windows,
  10 second hop, RMS >= -45 dB, at most 12 segments per song.
- One L2-normalized mean embedding per song.
- Balanced-prior Linear Discriminant Analysis with automatic covariance
  shrinkage.
- Grouped evaluation by `work_id` with 10 repeated stratified 5-fold runs.
- API deployment through the existing response schema, with KMeans retained
  only as an explicit fallback.
- A fresh checkout without the generated model starts in a diagnosable
  `degraded` health state; analysis remains unavailable until training runs.

## Data Result

- Input: 211 songs, 1466 legacy 30-second segment embeddings.
- Accepted: 174 independent canonical songs across 18 producers.
- Excluded: 37 songs, including 4 duplicate-work recordings.
- Rebuilt: 2086 MERT-v1-95M embeddings using the production segmentation.
- No accepted song is missing its source audio.
- Smallest classes: Neru and wowaka, 5 songs each.

Exclusion categories:

| Category | Songs |
|---|---:|
| Cover | 9 |
| Wrong entity | 7 |
| Collaboration | 6 |
| Game version | 5 |
| Duplicate work | 4 |
| Fan edit | 2 |
| Alternate vocal | 1 |
| Karaoke | 1 |
| Mashup | 1 |
| Style imitation | 1 |

## Final Evaluation

Protocol: 10 repeated `StratifiedGroupKFold` runs, 5 folds per repeat,
grouped by `work_id`, equal class priors.

| Metric | Mean | Repeat SD | Range |
|---|---:|---:|---:|
| Top-1 accuracy | 62.47% | 1.99% | 59.20%-66.09% |
| Top-3 accuracy | 79.48% | 2.25% | 76.44%-82.18% |
| Macro-F1 | 59.73% | 2.40% | 56.44%-63.66% |
| MRR | 72.94% | 1.49% | 71.21%-75.78% |

The earlier legacy 30-second embedding result was 66.78% Top-1. It is not
used as the final P0 number because its segmentation did not match the API.
LDA output scores are ranking scores and have not yet been probability
calibrated.

## Corrected Defects

1. Missing embedding files could shift labels because loaded vectors were
   zipped against the original records.
2. Legacy absolute paths pointed to a different checkout.
3. The audio segmenter omitted the final complete window.
4. MERT batch pooling multiplied a raw-sample attention mask by a
   frame-level hidden tensor and failed on every real batch.
5. Metadata and health endpoints described the old KMeans profiles even when
   a different retrieval model should serve requests.
6. Global pytest temporary storage is inaccessible on this Windows machine;
   tests now use a workspace-local temporary directory.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\07_curate_dataset.py
.\.venv\Scripts\python.exe scripts\07b_rebuild_curated_embeddings.py
.\.venv\Scripts\python.exe scripts\08_train_song_lda.py
.\.venv\Scripts\python.exe -m pytest -q
```

Generated artifacts:

- `data/processed/curated/mert_95/song_decisions.jsonl`
- `data/processed/curated/mert_95_p0_20s/segments.jsonl`
- `data/processed/evaluations/song_mean_shrinkage_lda.json`
- `data/processed/models/song_mean_shrinkage_lda.pkl`
