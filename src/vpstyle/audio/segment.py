"""Audio segmentation with RMS-based silence filtering."""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from vpstyle.data.metadata_schema import Segment
from vpstyle.utils.hashing import str_hash
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def rms_db(wav: np.ndarray) -> float:
    """Compute RMS energy of a waveform in dB."""
    rms = np.sqrt(np.mean(wav**2))
    if rms < 1e-8:
        return -100.0
    return float(20 * np.log10(rms))


def split_segments(
    wav: np.ndarray,
    sr: int,
    segment_seconds: float = 20.0,
    hop_seconds: float = 10.0,
    min_rms_db: float = -45.0,
    max_segments: int = 12,
) -> list[dict]:
    """Split waveform into overlapping segments, filtering low-energy ones.

    Returns a list of dicts with keys:
        start_sample, end_sample, start_sec, end_sec, duration_sec, rms_db
    """
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    total = len(wav)

    candidates: list[dict] = []
    for start in range(0, max(1, total - seg_len), hop_len):
        end = min(start + seg_len, total)
        chunk = wav[start:end]
        if len(chunk) < sr * 3:  # skip segments shorter than 3s
            continue
        rms = rms_db(chunk)
        candidates.append({
            "start_sample": start,
            "end_sample": end,
            "start_sec": start / sr,
            "end_sec": end / sr,
            "duration_sec": len(chunk) / sr,
            "rms_db": rms,
        })

    # Filter by RMS and keep top N
    valid = [c for c in candidates if c["rms_db"] >= min_rms_db]
    valid.sort(key=lambda c: c["rms_db"], reverse=True)
    selected = valid[:max_segments]

    logger.debug(
        "Segmentation: %d candidates, %d above threshold, %d selected",
        len(candidates), len(valid), len(selected),
    )
    return selected


def segment_file(
    wav_path: str | Path,
    output_dir: str | Path,
    producer_slug: str,
    song_id: str,
    sr: int = 24000,
    segment_seconds: float = 20.0,
    hop_seconds: float = 10.0,
    min_rms_db: float = -45.0,
    max_segments: int = 12,
) -> list[Segment]:
    """Segment a single preprocessed WAV file and save chunks to disk.

    Returns a list of Segment dataclasses.
    """
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    segments = split_segments(
        wav, sr,
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        min_rms_db=min_rms_db,
        max_segments=max_segments,
    )

    results: list[Segment] = []
    for seg in segments:
        chunk = wav[seg["start_sample"]:seg["end_sample"]]
        seg_id = str_hash(f"{song_id}_{seg['start_sec']:.1f}_{seg['end_sec']:.1f}")[:16]
        seg_path = output_dir / f"{seg_id}.wav"
        sf.write(str(seg_path), chunk, sr)

        segment = Segment(
            segment_id=seg_id,
            song_id=song_id,
            producer_slug=producer_slug,
            path=str(seg_path.resolve()),
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            duration_sec=seg["duration_sec"],
            rms_db=seg["rms_db"],
        )
        results.append(segment)

    logger.info(
        "Segmented %s: %d segments saved to %s",
        wav_path.name, len(results), output_dir,
    )
    return results


def save_segment_manifest(segments: list[Segment], path: str | Path) -> None:
    """Save segments as JSONL manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg.__dict__, ensure_ascii=False) + "\n")
    logger.info("Saved segment manifest: %s (%d segments)", path, len(segments))
