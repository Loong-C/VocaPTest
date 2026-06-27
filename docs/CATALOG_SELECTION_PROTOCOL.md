# Catalog Selection Protocol

This project treats catalog quality as part of the model, not as bookkeeping.
When adding producers, prefer fewer clean songs over a larger noisy set.

## Source Of Truth

- Use VocaDB API data as the primary evidence for producer identity, song type,
  artist roles, PVs, and tags.
- A catalog entry must include `vocadb_song_id`, `source_kind`, and either
  `source_service`/`source_id` or the legacy `youtube_id`.
- Supported media sources are VocaDB-listed enabled Original PVs from
  `Youtube` and `NicoNicoDouga`.
- Use VocaDB-listed `Other` or `Reprint` PVs only as explicit documented
  exceptions when an important canonical work lacks an enabled Original PV.
  Mark those entries with `source_kind: vocadb_other_pv` or
  `source_kind: vocadb_reprint`.
- Do not report VocaDB refreshes as successful unless the API returned JSON.
  If the request is blocked by Cloudflare, store exported API JSON locally and
  process that file instead.

## Inclusion Rules

- The VocaDB song must be `Original`.
- The target producer must be credited with `Composer` or `Default` roles.
- At least one voice-synth singer credit must be present, including Vocaloid,
  UTAU, CeVIO, SynthesizerV, Voiceroid, NewType, or OtherVoiceSynthesizer.
- Prefer higher VocaDB `ratingScore` entries, then manually review older
  representative works that may be important despite lower current ratings.
- Keep development and final holdouts representative; do not reserve only
  obscure tail songs for evaluation.
- Downloaded media must be between 60 and 600 seconds. Very long musicals,
  medleys, loop jams, or talk-heavy session videos should be replaced with
  cleaner representative works even when VocaDB marks them as Original PVs.

## Exclusion Rules

- Exclude covers, remixes, non-original PVs, disabled PVs, and entries where
  VocaDB does not support the producer/style attribution unless a reviewed
  sparse-catalog exception is documented in the catalog entry.
- Exclude songs with unapproved external producer or group style credits.
- Exclude songs that overlap another configured producer as a style credit.
- Avoid Topic/KARENT/SEGA/Project Sekai-style uploads when an original producer
  PV exists.
- Be cautious with commercial, contest, commissioned, collaboration, or later
  self-cover material when it is not representative of the producer's core
  Vocaloid style.

## Sparse Producers

- Do not skip a producer solely because the ideal 10 train / 2 dev / 4 final
  split cannot be filled.
- If the catalog is genuinely sparse after review, keep a smaller split:
  minimum 2 training songs and 1 final song; dev can be absent for that producer.
- Document the sparse decision in the expansion report, including which songs
  were reviewed and why the remainder was rejected.

## Split Safety

- Training, dev, and final splits must not share VocaDB song ids.
- Training, dev, and final splits must not share media source keys, where the
  source key is `youtube_<id>` or `niconico_<id>`.
- Materialized manifests must be regenerated after config edits so stale
  records are pruned.
- Final frozen results are for acceptance checks only; do not tune thresholds
  or model variants on final frozen feedback.

## Encoding Note

PowerShell may display Japanese UTF-8 text as mojibake even when the file is
correct. Verify suspected corruption with Python using `PYTHONIOENCODING=utf-8`
or by inspecting Unicode escapes before rewriting catalog titles.
