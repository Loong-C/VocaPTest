"""Build a feature store from embedding records."""
from __future__ import annotations

from pathlib import Path

from vocaptest.features.extract_embeddings import load_all_embeddings
from vocaptest.features.feature_store import FeatureStore
from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def build_feature_store(
    records: list[EmbeddingRecord],
    output_path: str | Path,
    build_faiss: bool = True,
) -> FeatureStore:
    """Load all embedding records and build a FeatureStore.

    Args:
        records: List of EmbeddingRecord.
        output_path: Directory to save the store.
        build_faiss: Whether to also build a FAISS index.

    Returns:
        The constructed FeatureStore.
    """
    embeddings, ids, slugs = load_all_embeddings(records)

    # Normalize for cosine similarity
    from sklearn.preprocessing import normalize
    embeddings = normalize(embeddings, norm="l2")

    store = FeatureStore()
    store.build(embeddings, ids, slugs)
    if build_faiss:
        store.build_faiss_index()
    store.save(output_path)

    logger.info(
        "FeatureStore built: %d embeddings, %d unique producers",
        len(ids),
        len(set(slugs)),
    )
    return store
