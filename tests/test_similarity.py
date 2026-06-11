"""Tests for similarity computation."""
import numpy as np
from vocaptest.retrieval.similarity import cosine_similarity, score_song_against_producer


def test_cosine_similarity_same():
    a = np.array([[1.0, 0.0, 0.0]])
    sim = cosine_similarity(a, a)
    assert abs(sim[0, 0] - 1.0) < 1e-4


def test_cosine_similarity_orthogonal():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    sim = cosine_similarity(a, b)
    assert abs(sim[0, 0]) < 1e-4


def test_score_song_against_producer():
    np.random.seed(42)
    song_embs = np.random.randn(8, 256).astype(np.float32)
    centroids = np.random.randn(5, 256).astype(np.float32)
    score = score_song_against_producer(song_embs, centroids, top_ratio=0.4)
    assert -1.0 <= score <= 1.0
    assert isinstance(score, float)


def test_cosine_similarity_1d():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    sim = cosine_similarity(a, b)
    assert abs(sim[0, 0] - 1.0) < 1e-4


def test_cosine_similarity_opposite_is_negative():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    sim = cosine_similarity(a, b)
    assert abs(sim[0, 0] + 1.0) < 1e-4
