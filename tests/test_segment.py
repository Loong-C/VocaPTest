"""Tests for audio segmentation."""
import numpy as np
from vocaptest.audio.segment import rms_db, split_segments


def test_rms_db_silence():
    wav = np.zeros(1000)
    assert rms_db(wav) == -100.0


def test_rms_db_normal():
    wav = np.ones(1000)
    db = rms_db(wav)
    assert -1.0 < db < 1.0  # near 0 dB


def test_split_segments_short_audio():
    sr = 24000
    wav = np.random.randn(sr * 5) * 0.1  # 5 seconds
    segments = split_segments(
        wav, sr,
        segment_seconds=20.0,
        hop_seconds=10.0,
        min_rms_db=-45.0,
        max_segments=12,
    )
    # Should return 0 or few segments since audio is shorter than segment length
    assert len(segments) <= 1


def test_split_segments_normal():
    sr = 24000
    wav = np.random.randn(sr * 60) * 0.1  # 60 seconds
    segments = split_segments(
        wav, sr,
        segment_seconds=20.0,
        hop_seconds=10.0,
        min_rms_db=-50.0,
        max_segments=12,
    )
    assert len(segments) > 0
    for seg in segments:
        assert seg["duration_sec"] <= 21.0
        assert seg["start_sec"] >= 0


def test_split_segments_includes_final_full_window():
    sr = 100
    wav = np.ones(sr * 60, dtype=np.float32)
    segments = split_segments(
        wav, sr,
        segment_seconds=20.0,
        hop_seconds=10.0,
        min_rms_db=-50.0,
        max_segments=12,
    )
    assert [segment["start_sec"] for segment in segments] == [0, 10, 20, 30, 40]


def test_split_segments_uniformly_covers_long_audio():
    sr = 10
    wav = np.ones(sr * 100, dtype=np.float32)
    segments = split_segments(
        wav,
        sr,
        segment_seconds=10,
        hop_seconds=10,
        min_rms_db=-50,
        max_segments=4,
        selection_strategy="uniform",
    )
    assert [segment["start_sec"] for segment in segments] == [0, 30, 60, 90]
