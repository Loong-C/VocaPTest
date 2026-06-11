import numpy as np

from vocaptest.features.song_features import mean_segment_embeddings
from vocaptest.models.song_lda import SongMeanShrinkageLDA


def test_song_mean_is_l2_normalized():
    segments = np.array([[3.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = mean_segment_embeddings(segments)
    assert np.allclose(result, [1.0, 0.0])
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_shrinkage_lda_ranks_song_mean():
    features = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9],
    ], dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    model = SongMeanShrinkageLDA.fit(
        features,
        labels,
        display_names={"a": "A", "b": "B", "c": "C"},
    )

    results = model.rank_segments(
        np.array([[0.95, 0.05, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        top_k=3,
    )

    assert results[0].producer_slug == "a"
    assert results[0].display_name == "A"
    assert [result.rank for result in results] == [1, 2, 3]
    assert all(0.0 <= result.score <= 1.0 for result in results)
