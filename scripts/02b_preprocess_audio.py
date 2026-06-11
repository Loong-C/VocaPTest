#!/usr/bin/env python
"""Preprocess audio: split into segments, save WAVs, compute RMS."""
import argparse
import json
import subprocess
import hashlib
from pathlib import Path

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def get_duration(filepath: Path) -> float:
    """Get audio duration in seconds using ffmprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def segment_audio(input_path: Path, output_dir: Path, song_id: str,
                  segment_duration: float = 30.0) -> list[dict]:
    """Split audio into fixed-length segments using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration = get_duration(input_path)
    if duration <= 0:
        return []

    segments = []
    seg_idx = 0
    
    start = 0.0
    while start < duration:
        end = min(start + segment_duration, duration)
        if end - start < 10:  # Skip very short last segment
            break

        seg_id = f"{song_id}_seg{seg_idx:03d}"
        out_path = output_dir / f"{seg_id}.wav"

        # Extract segment with ffmpeg
        result = subprocess.run([
            "ffmpeg", "-y", "-v", "quiet",
            "-ss", str(start), "-t", str(end - start),
            "-i", str(input_path),
            "-ac", "1", "-ar", "24000",  # mono, 24kHz for MERT
            str(out_path),
        ], capture_output=True, text=True, timeout=120)

        if result.returncode != 0 or not out_path.exists():
            logger.warning("ffmpeg segment failed: %s [%.1f-%.1f]", song_id, start, end)
        else:
            # Compute RMS using ffmpeg
            rms_result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-f", "lavfi",
                "-i", f"amovie={out_path},astats=metadata=1:reset=1",
                "-show_entries", "frame_tags=lavfi.astats.Overall.RMS_level",
                "-of", "default=noprint_wrappers=1:nokey=1",
            ], capture_output=True, text=True, timeout=30)
            
            try:
                rms_db = float(rms_result.stdout.strip().split("\n")[0])
            except (ValueError, IndexError):
                rms_db = -30.0  # default

            segments.append({
                "segment_id": seg_id,
                "song_id": song_id,
                "producer_slug": "",  # filled later
                "path": str(out_path.resolve()),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "rms_db": rms_db,
            })
            seg_idx += 1

        start = end

    return segments


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio into segments")
    parser.add_argument("--songs-jsonl", required=True, help="Path to songs metadata JSONL")
    parser.add_argument("--segments-dir", required=True, help="Output dir for segment WAV files")
    parser.add_argument("--manifest", required=True, help="Output path for segments JSONL manifest")
    parser.add_argument("--segment-duration", type=float, default=30.0, help="Duration per segment")
    args = parser.parse_args()

    with open(args.songs_jsonl, "r", encoding="utf-8") as f:
        songs = [json.loads(l) for l in f if l.strip()]

    segments_dir = Path(args.segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []
    skipped_no_audio = 0
    skipped_short = 0

    for song in songs:
        audio_path = song.get("local_audio_path")
        if not audio_path:
            skipped_no_audio += 1
            continue
        
        ap = Path(audio_path)
        if not ap.exists():
            skipped_no_audio += 1
            continue

        # Check if too short
        try:
            dur = get_duration(ap)
        except Exception:
            logger.warning("Cannot read duration for %s", ap)
            continue

        if dur < 30:
            skipped_short += 1
            # Still segment (will get 1 segment if >= 10s)
            if dur < 10:
                continue

        segs = segment_audio(ap, segments_dir, song["song_id"], args.segment_duration)
        for s in segs:
            s["producer_slug"] = song["producer_slug"]
        all_segments.extend(segs)

    # Save manifest
    with open(args.manifest, "w", encoding="utf-8") as f:
        for s in all_segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Segments: %d total, %d songs skipped (no audio), %d skipped (short)",
                 len(all_segments), skipped_no_audio, skipped_short)
    logger.info("Manifest saved to %s", args.manifest)


if __name__ == "__main__":
    main()
