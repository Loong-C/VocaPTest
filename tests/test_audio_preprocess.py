"""Tests for audio preprocessing."""
import pytest
import numpy as np
from vocaptest.audio.preprocess import normalize_peak, normalize_rms


def test_normalize_peak_silence():
    wav = np.zeros(1000)
    result = normalize_peak(wav, peak=0.95)
    assert np.allclose(result, wav)


def test_normalize_peak_normal():
    wav = np.array([0.0, 0.5, 1.0, -0.8])
    result = normalize_peak(wav, peak=0.95)
    assert np.max(np.abs(result)) <= 0.95
    assert np.max(np.abs(result)) >= 0.94


def test_normalize_rms_silence():
    wav = np.zeros(1000)
    result = normalize_rms(wav, target_db=-20.0)
    assert np.allclose(result, wav)


def test_normalize_rms_normal():
    wav = np.random.randn(10000) * 0.1
    result = normalize_rms(wav, target_db=-20.0)
    target_rms = 10 ** (-20.0 / 20.0)
    actual_rms = np.sqrt(np.mean(result ** 2))
    assert abs(actual_rms - target_rms) < 1e-3
