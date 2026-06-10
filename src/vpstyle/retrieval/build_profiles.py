"""Build producer profiles from segment embeddings."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

from vpstyle.data.metadata_schema import EmbeddingRecord
from vpstyle.features.extract_embeddings import load_all_embeddings
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def build_producer_profiles(
    records: list[EmbeddingRecord],
    num_clusters: int = 5,
    output_path: str | Path | None = None,
) -> dict:
    """Build KMeans-based producer profiles from segment embeddings.

    Args:
        records: List of EmbeddingRecord.
        num_clusters: Number of KMeans clusters per producer.
        output_path: If provided, save the profile dict as a pickle file.

    Returns:
        Dict with structure:
        {
            "backend": "mert_95",
            "producers": {
                "wowaka": {
                    "display_name": "wowaka",
                    "centroids": np.ndarray (num_clusters, dim),
                    "song_count": 30,
                    "segment_count": 240,
                },
                ...
            }
        }
    """
    embeddings, seg_ids, slugs = load_all_embeddings(records)
    embeddings = normalize(embeddings, norm="l2")

    # Group by producer
    producer_embs: dict[str, list[np.ndarray]] = {}
    producer_song_ids: dict[str, set[str]] = {}
    for emb, rec in zip(embeddings, records):
        slug = rec.producer_slug
        if slug not in producer_embs:
            producer_embs[slug] = []
            producer_song_ids[slug] = set()
        producer_embs[slug].append(emb)
        producer_song_ids[slug].add(rec.song_id)

    # Detect backend
    backend = records[0].model_backend if records else "unknown"

    # Build profiles
    profiles: dict = {"backend": backend, "producers": {}}
    for slug in tqdm(sorted(producer_embs.keys()), desc="Building profiles"):
        embs_arr = np.stack(producer_embs[slug], axis=0)
        n_segments = len(embs_arr)

        # Use fewer clusters if not enough samples
        n_clusters = min(num_clusters, n_segments)
        if n_clusters < 2:
            centroids = embs_arr.mean(axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embs_arr)
            centroids = normalize(kmeans.cluster_centers_, norm="l2")

        profiles["producers"][slug] = {
            "display_name": slug,
            "centroids": centroids,
            "song_count": len(producer_song_ids[slug]),
            "segment_count": n_segments,
        }

        logger.debug(
            "%s: %d songs, %d segments, %d centroids",
            slug,
            len(producer_song_ids[slug]),
            n_segments,
            centroids.shape[0],
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(profiles, f)
        logger.info("Producer profiles saved to %s", output_path)

    logger.info(
        "Built profiles for %d producers (backend=%s)",
        len(profiles["producers"]), backend,
    )
    return profiles


def load_profiles(path: str | Path) -> dict:
    """Load producer profiles from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
