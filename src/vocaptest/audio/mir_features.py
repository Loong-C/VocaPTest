"""Optional MIR feature extraction for debugging and validation."""
from __future__ import annotations

from typing import Optional

import librosa
import numpy as np

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def extract_mir_features(
    wav: np.ndarray,
    sr: int = 24000,
) -> dict[str, float]:
    """Extract traditional MIR features from a waveform.

    These features are NOT used for frontend display or main retrieval.
    They serve only as internal debugging tools to check whether the
    model is biased by non-stylistic factors (e.g. timbre, loudness).

    Returns a dict of scalar features.
    """
    features: dict[str, float] = {}

    # Spectral features
    S = np.abs(librosa.stft(wav))
    spec_centroid = librosa.feature.spectral_centroid(S=S)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=S)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(S=S)[0]

    features["spectral_centroid_mean"] = float(np.mean(spec_centroid))
    features["spectral_centroid_std"] = float(np.std(spec_centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spec_bandwidth))
    features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))

    # Rhythm
    try:
        tempo, _ = librosa.beat.beat_track(y=wav, sr=sr)
        features["bpm"] = float(tempo)
    except Exception:
        features["bpm"] = 0.0

    onset_env = librosa.onset.onset_strength(y=wav, sr=sr)
    features["onset_density"] = float(np.mean(onset_env))

    # Energy
    features["rms_db"] = float(20 * np.log10(np.sqrt(np.mean(wav**2)) + 1e-8))
    features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(wav)))

    # MFCC statistics
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return features


def extract_mir_features_file(
    wav_path: str | Path,
    sr: int = 24000,
) -> Optional[dict[str, float]]:
    """Load audio file and extract MIR features."""
    try:
        wav_path = str(wav_path)
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        return extract_mir_features(wav, sr=sr)
    except Exception as e:
        logger.error("MIR extraction failed for %s: %s", wav_path, e)
        return None
