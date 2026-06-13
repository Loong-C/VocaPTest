import numpy as np

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.features.layer_features import (
    build_song_segment_feature_tensor,
    pool_segment_layers,
)
from vocaptest.models.calibration import TemperatureScaler
from vocaptest.models.layer_fusion import LayerFusionLDA, optimize_nonnegative_weights
from vocaptest.models.song_lda import SongMeanShrinkageLDA


def test_pool_segment_layers_shapes_and_normalization():
    values = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2) + 1
    mean = pool_segment_layers(values, mode="mean")
    multi = pool_segment_layers(values, mode="mean_std_change")

    assert mean.shape == (3, 2)
    assert multi.shape == (3, 6)
    assert np.allclose(np.linalg.norm(mean, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(multi, axis=1), 1.0)


def test_nonnegative_layer_weights_prefer_better_layer():
    probabilities = np.array([
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
        [[0.4, 0.6], [0.6, 0.4], [0.6, 0.4], [0.4, 0.6]],
    ])
    labels = np.array([0, 0, 1, 1])
    weights = optimize_nonnegative_weights(probabilities, labels)

    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_selected_layer_model_uses_requested_hidden_layer():
    features = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=np.float32)
    labels = np.array(["a", "a", "b", "b"])
    estimator = SongMeanShrinkageLDA.fit(features, labels).estimator
    model = LayerFusionLDA(
        estimators=[estimator],
        layer_weights=np.array([1.0]),
        temperature_scaler=TemperatureScaler(1.0),
        rejection_threshold=0.0,
        layer_indices=[1],
        calibration_input="logits",
    )
    segment_layers = np.array([
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.9, 0.1], [0.0, 1.0]],
    ], dtype=np.float32)

    prediction = model.rank_segment_layers(segment_layers)

    assert prediction.results[0].producer_slug == "a"
    assert prediction.accepted


def test_build_song_segment_feature_tensor(monkeypatch):
    records = [
        EmbeddingRecord(
            segment_id="song_a_001",
            song_id="song_a",
            producer_slug="producer",
            model_backend="test",
            embedding_path="one.npy",
            embedding_dim=2,
            work_id="work_a",
            layer_count=2,
        ),
        EmbeddingRecord(
            segment_id="song_a_002",
            song_id="song_a",
            producer_slug="producer",
            model_backend="test",
            embedding_path="two.npy",
            embedding_dim=2,
            work_id="work_a",
            layer_count=2,
        ),
    ]
    arrays = {
        "one.npy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "two.npy": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    }
    monkeypatch.setattr(
        "vocaptest.features.layer_features.resolve_embedding_path",
        lambda record: record.embedding_path,
    )
    monkeypatch.setattr(
        "vocaptest.features.layer_features.np.load",
        lambda path: arrays[path],
    )

    features, masks, metadata = build_song_segment_feature_tensor(records, layer=1)

    assert features.shape == (1, 2, 2)
    assert np.array_equal(features[0], [[3.0, 4.0], [7.0, 8.0]])
    assert masks.tolist() == [[True, True]]
    assert metadata[0].work_id == "work_a"
