"""Compact traditional MIR features for composer-style ablations."""
from __future__ import annotations

import numpy as np


def mir_feature_names() -> list[str]:
    names = [
        "duration_seconds",
        "tempo_bpm",
        "onset_density_hz",
        "rms_mean",
        "rms_std",
        "rms_p10",
        "rms_p90",
        "rms_dynamic_range",
        "zcr_mean",
        "zcr_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "spectral_flatness_mean",
        "spectral_flatness_std",
        "harmonic_percussive_log_ratio",
        "chroma_peak_margin",
    ]
    names.extend(f"chroma_mean_{index}" for index in range(12))
    names.extend(f"chroma_std_{index}" for index in range(12))
    return names


def extract_mir_features(
    wav: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    duration = len(wav) / sample_rate
    onset_envelope = librosa.onset.onset_strength(
        y=wav,
        sr=sample_rate,
        hop_length=hop_length,
    )
    tempo = float(np.asarray(librosa.feature.tempo(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
    )).reshape(-1)[0])
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        units="time",
    )

    rms = librosa.feature.rms(y=wav, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(wav, hop_length=hop_length)[0]
    stft = np.abs(librosa.stft(wav, hop_length=hop_length))
    nyquist = sample_rate / 2.0
    centroid = librosa.feature.spectral_centroid(
        S=stft,
        sr=sample_rate,
    )[0] / nyquist
    bandwidth = librosa.feature.spectral_bandwidth(
        S=stft,
        sr=sample_rate,
    )[0] / nyquist
    rolloff = librosa.feature.spectral_rolloff(
        S=stft,
        sr=sample_rate,
    )[0] / nyquist
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    chroma = librosa.feature.chroma_stft(
        S=stft**2,
        sr=sample_rate,
        hop_length=hop_length,
    )
    harmonic, percussive = librosa.effects.hpss(wav)
    harmonic_energy = float(np.mean(harmonic**2))
    percussive_energy = float(np.mean(percussive**2))
    chroma_mean = chroma.mean(axis=1)
    sorted_chroma = np.sort(chroma_mean)

    features = np.concatenate([
        np.array([
            duration,
            tempo,
            len(onsets) / max(duration, 1e-6),
            rms.mean(),
            rms.std(),
            np.quantile(rms, 0.10),
            np.quantile(rms, 0.90),
            np.quantile(rms, 0.90) - np.quantile(rms, 0.10),
            zcr.mean(),
            zcr.std(),
            centroid.mean(),
            centroid.std(),
            bandwidth.mean(),
            bandwidth.std(),
            rolloff.mean(),
            rolloff.std(),
            flatness.mean(),
            flatness.std(),
            np.log10((harmonic_energy + 1e-10) / (percussive_energy + 1e-10)),
            sorted_chroma[-1] - sorted_chroma[-2],
        ]),
        chroma_mean,
        chroma.std(axis=1),
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
