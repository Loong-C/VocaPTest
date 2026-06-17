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
    wav = np.asarray(wav, dtype=np.float32)
    try:
        return _extract_mir_features_librosa(wav, sample_rate, hop_length)
    except ImportError:
        return _extract_mir_features_numpy(wav, sample_rate, hop_length)


def _extract_mir_features_librosa(
    wav: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    import librosa

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


def _extract_mir_features_numpy(
    wav: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    duration = len(wav) / sample_rate
    frames = _frame_signal(wav, frame_length=min(2048, max(256, len(wav))), hop_length=hop_length)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window

    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    power = spectrum**2
    freqs = np.fft.rfftfreq(windowed.shape[1], d=1.0 / sample_rate)
    nyquist = max(sample_rate / 2.0, 1e-6)
    freqs_norm = freqs / nyquist
    spectral_energy = spectrum.sum(axis=1) + 1e-12

    centroid = (spectrum * freqs_norm).sum(axis=1) / spectral_energy
    bandwidth = np.sqrt(
        (spectrum * (freqs_norm[None, :] - centroid[:, None]) ** 2).sum(axis=1)
        / spectral_energy
    )
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_threshold = 0.85 * spectral_energy
    rolloff_bins = (cumulative < rolloff_threshold[:, None]).sum(axis=1)
    rolloff = freqs_norm[np.clip(rolloff_bins, 0, len(freqs_norm) - 1)]
    flatness = np.exp(np.mean(np.log(spectrum + 1e-12), axis=1)) / (
        np.mean(spectrum + 1e-12, axis=1)
    )

    onset_envelope = np.maximum(np.diff(rms, prepend=rms[0]), 0.0)
    tempo = _estimate_tempo_bpm(onset_envelope, sample_rate, hop_length)
    onset_threshold = onset_envelope.mean() + onset_envelope.std()
    onsets = int(np.count_nonzero(onset_envelope > onset_threshold))

    chroma = _chroma_from_power(power, freqs)
    chroma_mean = chroma.mean(axis=1)
    chroma_std = chroma.std(axis=1)
    sorted_chroma = np.sort(chroma_mean)
    chroma_peak_margin = sorted_chroma[-1] - sorted_chroma[-2] if len(sorted_chroma) > 1 else 0.0

    low_energy = power[:, freqs <= 1000.0].mean() if np.any(freqs <= 1000.0) else 0.0
    high_energy = power[:, freqs > 1000.0].mean() if np.any(freqs > 1000.0) else 0.0

    features = np.concatenate([
        np.array([
            duration,
            tempo,
            onsets / max(duration, 1e-6),
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
            np.log10((low_energy + 1e-10) / (high_energy + 1e-10)),
            chroma_peak_margin,
        ]),
        chroma_mean,
        chroma_std,
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _frame_signal(
    wav: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if wav.size < frame_length:
        padded = np.pad(wav, (0, frame_length - wav.size))
        return padded.reshape(1, frame_length)

    starts = np.arange(0, wav.size - frame_length + 1, hop_length)
    if starts.size == 0 or starts[-1] != wav.size - frame_length:
        starts = np.append(starts, wav.size - frame_length)
    return np.stack([wav[start:start + frame_length] for start in starts]).astype(np.float32)


def _estimate_tempo_bpm(
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> float:
    if onset_envelope.size < 3 or np.allclose(onset_envelope, 0.0):
        return 0.0
    centered = onset_envelope - onset_envelope.mean()
    autocorr = np.correlate(centered, centered, mode="full")[centered.size - 1:]
    min_lag = max(1, int((60.0 / 240.0) * sample_rate / hop_length))
    max_lag = min(len(autocorr), int((60.0 / 40.0) * sample_rate / hop_length) + 1)
    if max_lag <= min_lag:
        return 0.0
    lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag]))
    return float(60.0 * sample_rate / (lag * hop_length))


def _chroma_from_power(
    power: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    chroma = np.zeros((12, power.shape[0]), dtype=np.float64)
    valid = freqs > 0.0
    if not np.any(valid):
        return chroma.astype(np.float32)

    midi = np.rint(12.0 * np.log2(freqs[valid] / 440.0) + 69.0).astype(int)
    pitch_classes = np.mod(midi, 12)
    for pitch_class in range(12):
        bins = valid.copy()
        bins[valid] = pitch_classes == pitch_class
        chroma[pitch_class] = power[:, bins].sum(axis=1)
    chroma /= chroma.sum(axis=0, keepdims=True) + 1e-12
    return chroma.astype(np.float32)
