"""Audio download via yt-dlp."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from vocaptest.data.metadata_schema import AudioManifestEntry, Song
from vocaptest.utils.hashing import file_hash
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def download_audio(
    url: str,
    output_dir: str | Path,
    output_template: str = "%(id)s",
    audio_format: str = "wav",
) -> Optional[Path]:
    """Download audio from a URL using yt-dlp. Returns path to downloaded file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / f"{output_template}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", "0",
        "-o", out_template,
        "--no-playlist",
        "--no-overwrites",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("yt-dlp failed for %s: %s", url, result.stderr[:500])
            return None
    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out for %s", url)
        return None
    except FileNotFoundError:
        logger.error(
            "yt-dlp not found. Install with: pip install yt-dlp"
        )
        return None

    # Find the downloaded file
    downloaded = sorted(output_dir.glob(f"{output_template}.*"))
    if not downloaded:
        logger.warning("No output file found for %s", url)
        return None
    return downloaded[-1]


def build_audio_manifest(
    songs: list[Song],
    audio_dir: str | Path,
    manifest_path: str | Path,
) -> list[AudioManifestEntry]:
    """Scan downloaded audio files and produce a manifest."""
    import subprocess as sp

    audio_dir = Path(audio_dir)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["song_id"])

    entries: list[AudioManifestEntry] = []
    for song in songs:
        if song.status != "accepted" or not song.local_audio_path:
            continue
        if song.song_id in existing:
            continue

        fpath = Path(song.local_audio_path)
        if not fpath.exists():
            continue

        # Probe audio info via ffprobe
        duration = 0.0
        sample_rate = 0
        channels = 0
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(fpath),
            ]
            probe = sp.run(probe_cmd, capture_output=True, text=True, timeout=30)
            if probe.returncode == 0:
                info = json.loads(probe.stdout)
                fmt = info.get("format", {})
                duration = float(fmt.get("duration", 0))
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        sample_rate = int(stream.get("sample_rate", 0))
                        channels = int(stream.get("channels", 0))
                        break
        except Exception:
            logger.warning("Failed to probe %s", fpath)

        entry = AudioManifestEntry(
            file_hash=file_hash(fpath),
            path=str(fpath.resolve()),
            duration_sec=duration,
            sample_rate=sample_rate,
            channels=channels,
            source_url=song.source_urls[0] if song.source_urls else "",
            producer_slug=song.producer_slug,
            song_id=song.song_id,
        )
        entries.append(entry)

    # Append to manifest
    with open(manifest_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")

    logger.info("Added %d entries to audio manifest", len(entries))
    return entries
