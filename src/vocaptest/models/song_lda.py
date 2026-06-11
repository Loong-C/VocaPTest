"""Song-mean Linear Discriminant Analysis with automatic covariance shrinkage."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from vocaptest.data.metadata_schema import SearchResult
from vocaptest.features.song_features import mean_segment_embeddings


class SongMeanShrinkageLDA:
    """Balanced-prior Shrinkage LDA trained on one vector per song."""

    format_version = 1

    def __init__(
        self,
        estimator: LinearDiscriminantAnalysis,
        display_names: dict[str, str] | None = None,
        catalog: dict | None = None,
        embedding_backend: str = "unknown",
    ):
        self.estimator = estimator
        self.display_names = display_names or {}
        self.catalog = catalog or {}
        self.embedding_backend = embedding_backend

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: list[str] | np.ndarray,
        display_names: dict[str, str] | None = None,
        catalog: dict | None = None,
        embedding_backend: str = "unknown",
    ) -> "SongMeanShrinkageLDA":
        labels = np.asarray(labels)
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("Shrinkage LDA requires at least two producer classes")
        priors = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=priors,
        )
        estimator.fit(features, labels)
        return cls(estimator, display_names, catalog, embedding_backend)

    @property
    def classes_(self) -> np.ndarray:
        return self.estimator.classes_

    def predict_proba(self, song_features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(song_features)

    def rank_segments(
        self,
        segment_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        song_vector = mean_segment_embeddings(segment_embeddings)
        probabilities = self.predict_proba(song_vector[None, :])[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        return [
            SearchResult(
                producer_slug=str(self.classes_[index]),
                display_name=self.display_names.get(
                    str(self.classes_[index]), str(self.classes_[index])
                ),
                score=float(probabilities[index]),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]

    def to_reference_library(self) -> dict:
        producers = {}
        counts = self.catalog.get("song_counts", {})
        segment_counts = self.catalog.get("segment_counts", {})
        for slug in self.classes_:
            slug = str(slug)
            producers[slug] = {
                "display_name": self.display_names.get(slug, slug),
                "song_count": counts.get(slug),
                "segment_count": segment_counts.get(slug),
            }
        return {
            "backend": f"{self.embedding_backend}+song_mean_shrinkage_lda",
            "producers": producers,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "format_version": self.format_version,
            "estimator": self.estimator,
            "display_names": self.display_names,
            "catalog": self.catalog,
            "embedding_backend": self.embedding_backend,
        }
        with open(path, "wb") as handle:
            pickle.dump(artifact, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SongMeanShrinkageLDA":
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        if artifact.get("format_version") != cls.format_version:
            raise ValueError("Unsupported SongMeanShrinkageLDA artifact version")
        return cls(
            estimator=artifact["estimator"],
            display_names=artifact.get("display_names"),
            catalog=artifact.get("catalog"),
            embedding_backend=artifact.get("embedding_backend", "unknown"),
        )


def build_catalog(labels: list[str], segment_counts: list[int]) -> dict:
    song_counts: Counter[str] = Counter()
    total_segments: Counter[str] = Counter()
    for label, count in zip(labels, segment_counts):
        song_counts[label] += 1
        total_segments[label] += count
    return {
        "song_counts": dict(sorted(song_counts.items())),
        "segment_counts": dict(sorted(total_segments.items())),
    }
