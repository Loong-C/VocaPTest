"""Validate dataset integrity — check files exist, manifest consistency."""
from __future__ import annotations

import json
from pathlib import Path

from vpstyle.data.metadata_schema import AudioManifestEntry
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def validate_manifest(manifest_path: str | Path) -> dict:
    """Validate audio manifest. Returns a summary dict."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {"status": "missing", "total": 0, "missing_files": 0, "errors": []}

    entries: list[AudioManifestEntry] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(AudioManifestEntry(**json.loads(line)))

    missing = [e for e in entries if not Path(e.path).exists()]
    producers: dict[str, int] = {}
    for e in entries:
        producers[e.producer_slug] = producers.get(e.producer_slug, 0) + 1

    summary = {
        "status": "ok" if not missing else "incomplete",
        "total": len(entries),
        "missing_files": len(missing),
        "missing_details": [
            {"song_id": e.song_id, "path": e.path} for e in missing[:20]
        ],
        "producer_counts": producers,
    }

    logger.info("Manifest validation: %s", json.dumps(summary, indent=2))
    return summary


def check_song_balance(song_index_path: str | Path, min_songs: int = 10) -> bool:
    """Check that every producer has enough accepted songs."""
    song_index_path = Path(song_index_path)
    if not song_index_path.exists():
        logger.warning("Song index not found: %s", song_index_path)
        return False

    producer_counts: dict[str, int] = {}
    with open(song_index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") == "accepted":
                slug = s.get("producer_slug", "unknown")
                producer_counts[slug] = producer_counts.get(slug, 0) + 1

    all_ok = True
    for slug, count in sorted(producer_counts.items()):
        status = "✅" if count >= min_songs else "❌"
        logger.info("%s %s: %d songs", status, slug, count)
        if count < min_songs:
            all_ok = False

    return all_ok
