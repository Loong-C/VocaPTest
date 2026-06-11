"""Feature store for efficient similarity search."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


class FeatureStore:
    """Lightweight feature store backed by numpy arrays and (optionally) FAISS."""

    def __init__(self):
        self._embeddings: Optional[np.ndarray] = None
        self._ids: list[str] = []
        self._producer_slugs: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._index = None  # FAISS index, built lazily

    @property
    def dim(self) -> int:
        if self._embeddings is None:
            return 0
        return self._embeddings.shape[1]

    def build(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        producer_slugs: list[str],
    ) -> None:
        """Initialize the store with embeddings."""
        self._embeddings = embeddings.astype(np.float32)
        self._ids = list(ids)
        self._producer_slugs = list(producer_slugs)
        self._id_to_idx = {sid: i for i, sid in enumerate(ids)}
        logger.info("FeatureStore built: %d vectors, dim=%d", len(ids), embeddings.shape[1])

    def build_faiss_index(self) -> None:
        """Build a FAISS IndexFlatIP for cosine similarity search."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed — using numpy for search")
            return

        if self._embeddings is None:
            return
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(self._embeddings)
        logger.info("FAISS index built: %d vectors", self._index.ntotal)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query: (dim,) or (batch, dim) query vector(s).
            k: Number of results.

        Returns:
            List of (segment_id, score) tuples for single query,
            or list of lists for batch queries.
        """
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]

        if self._index is not None:
            scores, indices = self._index.search(query, k)
            results = []
            for i in range(len(query)):
                batch_results = [
                    (self._ids[idx], float(scores[i][j]))
                    for j, idx in enumerate(indices[i])
                    if idx >= 0
                ]
                results.append(batch_results)
        else:
            # Numpy fallback: cosine similarity = dot product (both normalized)
            sims = np.dot(query, self._embeddings.T)
            results = []
            for i in range(len(query)):
                top_idx = np.argsort(sims[i])[-k:][::-1]
                batch_results = [
                    (self._ids[idx], float(sims[i][idx]))
                    for idx in top_idx
                ]
                results.append(batch_results)

        if len(results) == 1:
            return results[0]
        return results

    def get_embedding(self, segment_id: str) -> Optional[np.ndarray]:
        """Get embedding by segment ID."""
        idx = self._id_to_idx.get(segment_id)
        if idx is None or self._embeddings is None:
            return None
        return self._embeddings[idx]

    def save(self, path: str | Path) -> None:
        """Save the store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(str(path / "embeddings.npy"), self._embeddings)
        with open(path / "ids.txt", "w", encoding="utf-8") as f:
            for sid in self._ids:
                f.write(sid + "\n")
        with open(path / "slugs.txt", "w", encoding="utf-8") as f:
            for slug in self._producer_slugs:
                f.write(slug + "\n")
        logger.info("FeatureStore saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStore":
        """Load the store from disk."""
        path = Path(path)
        store = cls()
        store._embeddings = np.load(str(path / "embeddings.npy"))
        with open(path / "ids.txt", "r", encoding="utf-8") as f:
            store._ids = [line.strip() for line in f]
        with open(path / "slugs.txt", "r", encoding="utf-8") as f:
            store._producer_slugs = [line.strip() for line in f]
        store._id_to_idx = {sid: i for i, sid in enumerate(store._ids)}
        logger.info("FeatureStore loaded from %s: %d vectors", path, len(store._ids))
        return store
