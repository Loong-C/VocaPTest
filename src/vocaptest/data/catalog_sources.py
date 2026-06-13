"""Validation and download helpers for vetted YouTube/VocaDB catalogs."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


VOCADB_API = "https://vocadb.net/api"


def yt_dlp_command(root: Path) -> list[str]:
    bundled = root / "tools" / "yt-dlp.exe"
    if bundled.exists():
        return [str(bundled)]
    return [sys.executable, "-m", "yt_dlp"]


def yt_dlp_runtime_args() -> list[str]:
    args = ["--remote-components", "ejs:github"]
    node = shutil.which("node")
    if node:
        args.extend(["--js-runtimes", f"node:{node}"])
    return args


def read_youtube_metadata(command: list[str], url: str) -> dict:
    result = subprocess.run(
        command
        + yt_dlp_runtime_args()
        + ["--skip-download", "--no-warnings", "--dump-single-json", url],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Metadata failed: {url}")
    return json.loads(result.stdout)


def download_youtube_audio(
    command: list[str],
    url: str,
    output_path: Path,
    ffmpeg_location: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_args = (
        ["--ffmpeg-location", str(ffmpeg_location)]
        if ffmpeg_location
        else []
    )
    result = None
    for attempt in range(3):
        result = subprocess.run(
            command
            + yt_dlp_runtime_args()
            + [
                "-f",
                "bestaudio[ext=m4a]/bestaudio",
                "--extract-audio",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "192K",
                "--no-playlist",
                "--no-warnings",
                "--retries",
                "5",
                "--fragment-retries",
                "5",
            ]
            + ffmpeg_args
            + [
                "-o",
                str(output_path.with_suffix(".%(ext)s")),
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=420,
            check=False,
        )
        if result.returncode == 0 and output_path.exists():
            return
        if attempt < 2:
            time.sleep(2 ** attempt)
    assert result is not None
    detail = result.stderr.strip().splitlines()
    raise RuntimeError(
        detail[-1] if detail else f"Audio download failed: {url}"
    )


def validate_vocadb_original(
    *,
    song_id: int,
    youtube_id: str,
    artist_id: int,
    session: requests.Session,
    allowed_pv_types: tuple[str, ...] = ("Original",),
) -> dict:
    response = None
    for attempt in range(3):
        try:
            response = session.get(
                f"{VOCADB_API}/songs/{song_id}",
                params={"fields": "Artists,PVs"},
                timeout=60,
            )
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    assert response is not None
    response.raise_for_status()
    song = response.json()
    if song.get("songType") != "Original":
        raise ValueError(
            f"VocaDB song {song_id} is {song.get('songType')}, not Original"
        )

    credits = [
        credit
        for credit in song.get("artists", [])
        if credit.get("artist", {}).get("id") == artist_id
    ]
    roles = {
        role.strip()
        for credit in credits
        for role in credit.get("effectiveRoles", "").split(",")
        if role.strip()
    }
    if not ({"Composer", "Default"} & roles):
        raise ValueError(
            f"VocaDB song {song_id} does not credit artist {artist_id} "
            f"as composer: {sorted(roles)}"
        )

    matching_pvs = [
        pv
        for pv in song.get("pvs", [])
        if pv.get("service") == "Youtube"
        and pv.get("pvType") in allowed_pv_types
        and not pv.get("disabled")
        and pv.get("pvId") == youtube_id
    ]
    if not matching_pvs:
        raise ValueError(
            f"YouTube {youtube_id} is not an enabled "
            f"{'/'.join(allowed_pv_types)} PV for VocaDB song {song_id}"
        )
    return {
        "vocadb_name": song.get("name"),
        "vocadb_artist_roles": sorted(roles),
        "vocadb_pv_author": matching_pvs[0].get("author"),
        "vocadb_pv_type": matching_pvs[0].get("pvType"),
    }


def source_reason(source_kind: str) -> str:
    reasons = {
        "official_upload": (
            "VocaDB Original work with a verified official YouTube upload"
        ),
        "vocadb_original_pv": (
            "VocaDB Original work with a VocaDB-listed Original YouTube PV"
        ),
        "vocadb_reprint": (
            "VocaDB Original work with a VocaDB-listed YouTube reprint"
        ),
    }
    try:
        return reasons[source_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported source_kind: {source_kind}") from exc
