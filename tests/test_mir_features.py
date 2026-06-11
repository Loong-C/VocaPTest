import numpy as np

from vocaptest.features.mir_features import extract_mir_features, mir_feature_names


def test_mir_features_are_finite_and_stable_shape():
    sample_rate = 8000
    time = np.arange(sample_rate * 2) / sample_rate
    wav = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    features = extract_mir_features(wav, sample_rate)

    assert features.shape == (len(mir_feature_names()),)
    assert np.all(np.isfinite(features))
