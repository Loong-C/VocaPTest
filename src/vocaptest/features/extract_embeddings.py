"""Batch embedding extraction with caching."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vocaptest.data.metadata_schema import EmbeddingRecord, Segment
from vocaptest.models.base import AudioEmbedder
from vocaptest.utils.logging import setup_logging
from vocaptest.utils.paths import project_root

logger = setup_logging()


def extract_embeddings(
    segments: list[Segment],
    embedder: AudioEmbedder,
    output_dir: str | Path,
    manifest_path: str | Path,
    resume: bool = True,
) -> list[EmbeddingRecord]:
    """Extract embeddings for all segments, caching results to disk.

    Args:
        segments: List of Segment dataclasses.
        embedder: An AudioEmbedder instance.
        output_dir: Directory to save .npy embedding files.
        manifest_path: Path to save the embedding manifest JSONL.
        resume: If True, skip segments that already have an embedding file.

    Returns:
        List of EmbeddingRecord dataclasses.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing manifest for resume
    existing_ids: set[str] = set()
    if resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["segment_id"])

    records: list[EmbeddingRecord] = []
    skipped = 0
    failed = 0

    for seg in tqdm(segments, desc="Extracting embeddings"):
        if seg.segment_id in existing_ids:
            npy_path = output_dir / f"{seg.segment_id}.npy"
            if npy_path.exists():
                skipped += 1
                records.append(EmbeddingRecord(
                    segment_id=seg.segment_id,
                    song_id=seg.song_id,
                    producer_slug=seg.producer_slug,
                    model_backend=embedder.backend_name,
                    embedding_path=str(npy_path.resolve()),
                    embedding_dim=embedder.dim,
                ))
                continue

        seg_path = Path(seg.path)
        if not seg_path.exists():
            logger.warning("Segment file not found: %s", seg.path)
            failed += 1
            continue

        try:
            t0 = time.time()
            emb = embedder.embed_file(str(seg_path))
            elapsed = time.time() - t0

            npy_path = output_dir / f"{seg.segment_id}.npy"
            np.save(str(npy_path), emb)

            record = EmbeddingRecord(
                segment_id=seg.segment_id,
                song_id=seg.song_id,
                producer_slug=seg.producer_slug,
                model_backend=embedder.backend_name,
                embedding_path=str(npy_path.resolve()),
                embedding_dim=embedder.dim,
            )
            records.append(record)
            existing_ids.add(seg.segment_id)
        except Exception as e:
            logger.error("Failed to embed %s: %s", seg.segment_id, e)
            failed += 1

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        seen: set[str] = set()
        for rec in records:
            if rec.segment_id not in seen:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                seen.add(rec.segment_id)

    logger.info(
        "Embedding extraction complete: %d extracted, %d skipped, %d failed",
        len(records) - skipped, skipped, failed,
    )
    return records


def load_embedding(record: EmbeddingRecord) -> np.ndarray:
    """Load a single embedding from disk."""
    return np.load(resolve_embedding_path(record))


def resolve_embedding_path(record: EmbeddingRecord) -> Path:
    """Resolve both current relative paths and legacy absolute manifest paths."""
    root = project_root()
    stored = Path(record.embedding_path)
    candidates = [stored]
    if not stored.is_absolute():
        candidates.extend([root / stored, Path.cwd() / stored])
    candidates.append(
        root
        / "data"
        / "processed"
        / "embeddings"
        / record.model_backend
        / f"{record.segment_id}.npy"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Embedding not found for {record.segment_id}; tried: {tried}")


def load_all_embeddings_aligned(
    records: list[EmbeddingRecord],
) -> tuple[np.ndarray, list[EmbeddingRecord]]:
    """Load embeddings and return only the records aligned with loaded rows."""
    embs: list[np.ndarray] = []
    loaded_records: list[EmbeddingRecord] = []
    for rec in tqdm(records, desc="Loading embeddings"):
        try:
            embs.append(load_embedding(rec))
            loaded_records.append(rec)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", rec.segment_id, exc)

    if not embs:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(embs, axis=0), loaded_records


def load_all_embeddings(
    records: list[EmbeddingRecord],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load all embeddings into a single array.

    Returns:
        (embeddings_array, segment_ids, producer_slugs)
    """
    embeddings, loaded_records = load_all_embeddings_aligned(records)
    return (
        embeddings,
        [record.segment_id for record in loaded_records],
        [record.producer_slug for record in loaded_records],
    )
