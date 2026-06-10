"""Batch embedding extraction with caching."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vpstyle.data.metadata_schema import EmbeddingRecord, Segment
from vpstyle.models.base import AudioEmbedder
from vpstyle.utils.logging import setup_logging

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
    return np.load(record.embedding_path)


def load_all_embeddings(
    records: list[EmbeddingRecord],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load all embeddings into a single array.

    Returns:
        (embeddings_array, segment_ids, producer_slugs)
    """
    embs = []
    seg_ids = []
    slugs = []
    for rec in tqdm(records, desc="Loading embeddings"):
        try:
            emb = load_embedding(rec)
            embs.append(emb)
            seg_ids.append(rec.segment_id)
            slugs.append(rec.producer_slug)
        except Exception as e:
            logger.warning("Failed to load %s: %s", rec.segment_id, e)
    return np.stack(embs, axis=0), seg_ids, slugs
