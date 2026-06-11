import numpy as np

from vocaptest.models.calibration import (
    TemperatureScaler,
    confidence_signals,
    select_rejection_threshold,
)


def test_temperature_scaler_returns_probabilities():
    probabilities = np.array([
        [0.99, 0.01],
        [0.95, 0.05],
        [0.90, 0.10],
        [0.80, 0.20],
    ])
    labels = np.array([0, 1, 0, 1])
    scaler = TemperatureScaler.fit(probabilities, labels)
    calibrated = scaler.transform(probabilities)

    assert scaler.temperature > 0
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_temperature_scaler_accepts_unsaturated_logits():
    logits = np.array([
        [8.0, 1.0],
        [3.0, 2.0],
        [1.0, 5.0],
        [2.0, 2.5],
    ])
    labels = np.array([0, 1, 1, 0])
    scaler = TemperatureScaler.fit_logits(logits, labels)
    calibrated = scaler.transform_logits(logits)

    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert calibrated[0, 0] != calibrated[1, 0]


def test_confidence_signals_and_rejection_threshold():
    probabilities = np.array([
        [0.90, 0.10],
        [0.80, 0.20],
        [0.55, 0.45],
        [0.51, 0.49],
    ])
    labels = np.array([0, 0, 1, 1])
    signals = confidence_signals(probabilities)
    threshold = select_rejection_threshold(
        probabilities,
        labels,
        target_precision=1.0,
        minimum_coverage=0.25,
    )

    assert np.allclose(signals["margin"], [0.8, 0.6, 0.1, 0.02])
    assert threshold["precision"] == 1.0
    assert threshold["coverage"] == 0.5
