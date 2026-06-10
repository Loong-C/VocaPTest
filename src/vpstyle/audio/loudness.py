"""Loudness analysis utilities."""
from __future__ import annotations

import numpy as np

from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def loudness_rms_db(wav: np.ndarray) -> float:
    """Compute RMS-based loudness in dB."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-8:
        return -100.0
    return float(20 * np.log10(rms))


def loudness_integrated_lufs(wav: np.ndarray, sr: int = 24000) -> float:
    """Estimate integrated LUFS using a simple RMS approximation.

    For production use, replace with pyloudnorm or ffmpeg loudnorm.
    """
    # Simple approximation — real LUFS needs gating
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-8:
        return -70.0
    return float(20 * np.log10(rms)) - 0.691  # approx offset from dB FS to LUFS


def segment_loudness_profile(
    wav: np.ndarray,
    sr: int,
    window_seconds: float = 1.0,
) -> list[float]:
    """Sliding window loudness profile in dB."""
    window_samples = int(window_seconds * sr)
    profile: list[float] = []
    for start in range(0, len(wav) - window_samples, window_samples):
        chunk = wav[start:start + window_samples]
        profile.append(loudness_rms_db(chunk))
    return profile
