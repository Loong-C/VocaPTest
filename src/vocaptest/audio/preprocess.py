"""Audio preprocessing: load, normalize, resample, save."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def load_audio(path: str | Path, sr: int = 24000, mono: bool = True) -> np.ndarray:
    """Load audio file, resample, and optionally convert to mono."""
    wav, in_sr = sf.read(str(path), dtype='float32')
    if mono and wav.ndim > 1:
        wav = wav.mean(axis=1)
    if in_sr != sr:
        num_samples = int(len(wav) * sr / in_sr)
        wav = scipy_resample(wav, num_samples)
    return wav


def normalize_peak(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize waveform to a target peak amplitude."""
    max_abs = np.max(np.abs(wav))
    if max_abs < 1e-8:
        return wav
    return wav / max_abs * peak


def normalize_rms(wav: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS normalize to a target dB level."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-8:
        return wav
    target_rms = 10 ** (target_db / 20.0)
    return wav * (target_rms / rms)


def save_wav(wav: np.ndarray, path: str | Path, sr: int = 24000) -> None:
    """Save waveform as WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    sr: int = 24000,
    mono: bool = True,
    normalize: str = "peak",
    peak: float = 0.95,
) -> np.ndarray:
    """Load, preprocess, and save a single audio file.

    Returns the processed waveform.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    wav = load_audio(input_path, sr=sr, mono=mono)

    if normalize == "peak":
        wav = normalize_peak(wav, peak=peak)
    elif normalize == "rms":
        wav = normalize_rms(wav)

    save_wav(wav, output_path, sr=sr)
    duration = len(wav) / sr
    logger.info("Preprocessed: %s -> %s (%.1fs)", input_path.name, output_path.name, duration)
    return wav


def batch_preprocess(
    input_dir: str | Path,
    output_dir: str | Path,
    sr: int = 24000,
    mono: bool = True,
    normalize: str = "peak",
) -> list[Path]:
    """Preprocess all WAV files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed: list[Path] = []
    for fpath in sorted(input_dir.glob("*.wav")):
        out_path = output_dir / fpath.name
        try:
            preprocess_file(fpath, out_path, sr=sr, mono=mono, normalize=normalize)
            processed.append(out_path)
        except Exception as e:
            logger.error("Failed to process %s: %s", fpath, e)

    logger.info("Batch preprocess complete: %d/%d files", len(processed),
                 len(list(input_dir.glob("*.wav"))))
    return processed
