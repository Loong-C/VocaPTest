#!/usr/bin/env python
"""Refresh cached producer style tags from VocaDB song tag data.

The deployed app reads configs/producer_style_tags.yaml and never calls VocaDB
at runtime. This script is for periodic offline refreshes from either:

1. VocaDB API access, when the environment is allowed through Cloudflare.
2. Browser-exported raw JSONL files in data/raw_jsonl/<slug>_songs.jsonl.

The script keeps existing tags when no usable source data is available, so a
blocked network refresh cannot accidentally blank the UI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocaptest.data.vocadb_client import VocaDBClient


DEFAULT_OUTPUT = Path("configs/producer_style_tags.yaml")

TAG_TRANSLATIONS = {
    "acoustic": "原声",
    "alternative rock": "另类摇滚",
    "ballad": "Ballad",
    "bitpop": "Bitpop",
    "breakcore": "Breakcore",
    "chiptune": "Chiptune",
    "city pop": "City Pop",
    "classical": "古典",
    "dance": "Dance",
    "dance-pop": "舞曲流行",
    "dance-punk": "Dance Punk",
    "dark": "暗色",
    "denpa song": "电波",
    "digital rock": "Digital Rock",
    "drum": "鼓",
    "drum and bass": "Drum and Bass",
    "dubstep": "Dubstep",
    "edm": "EDM",
    "electronica": "Electronica",
    "electric guitar": "电吉他",
    "electronic": "电子",
    "electropop": "电子流行",
    "experimental": "实验",
    "folk": "民谣",
    "funk rock": "Funk Rock",
    "gothic rock (japanese)": "Gothic Rock",
    "guitar": "吉他",
    "high tempo": "高BPM",
    "house": "House",
    "j-pop": "J-Pop",
    "j-rock": "J-Rock",
    "jazz": "Jazz",
    "kawaii future bass": "Kawaii Future Bass",
    "math rock": "Math Rock",
    "melancholy": "Melancholy",
    "melancholic": "忧郁",
    "metal": "Metal",
    "minimal music": "Minimal",
    "minimal": "Minimal",
    "orchestra": "管弦",
    "orchestral": "管弦",
    "piano": "钢琴",
    "pop": "流行",
    "pop punk": "Pop Punk",
    "pop rock": "Pop Rock",
    "post-rock": "Post-Rock",
    "progressive": "Progressive",
    "punk rock": "Punk Rock",
    "rap": "Rap",
    "rock": "摇滚",
    "sad": "悲伤",
    "satire": "讽刺",
    "shoegaze": "Shoegaze",
    "story": "叙事",
    "stylish": "时髦",
    "summer": "夏日",
    "synthpop": "Synthpop",
    "techno": "Techno",
    "technopop": "Technopop",
    "trap music": "Trap",
    "trance": "Trance",
    "電波": "电波",
    "和風": "和风",
    "夜好性": "夜好性",
    "traditional japanese": "和风",
    "vocarock": "VOCAROCK",
}

STYLE_CATEGORY_PRIORITY = {
    "Genres": 0,
    "Instruments": 1,
    "Composition": 2,
    "Subjective": 3,
    "Themes": 4,
}

NON_STYLE_CATEGORIES = {
    "Animation",
    "Copyrights",
    "Distribution",
    "Editor notes",
    "Events",
    "Games",
    "Lyrics",
    "Sources",
    "Vocalists",
}

NON_STYLE_EXACT = {
    "album",
    "album-exclusive song",
    "anime",
    "cevio ai",
    "cevio",
    "color",
    "cover",
    "duet",
    "english",
    "fanmade",
    "featured",
    "female vocal",
    "free",
    "game edit",
    "good tuning",
    "gumi",
    "hatsune miku",
    "human singers",
    "instrumental",
    "japanese",
    "kagamine len",
    "kagamine rin",
    "kaito",
    "karaoke",
    "kasane teto",
    "live",
    "lyrics",
    "magical mirai",
    "meiko",
    "megurine luka",
    "miku",
    "music video",
    "nico nico douga",
    "original song",
    "other vocals",
    "pv",
    "remix",
    "self-cover",
    "self-remix",
    "song contest",
    "synthesizer v",
    "translation request",
    "utau",
    "vocaloid",
    "vocaloid 2",
    "vocaloid 3",
    "vocaloid 4",
    "vocaloid 5",
    "vocaloid original",
    "youtube",
    "youtube premium",
}

NON_STYLE_SUBSTRINGS = (
    "album exclusive",
    "concert",
    "contest",
    "event",
    "feat.",
    "festival",
    "game size",
    "karaoke",
    "lyrics:",
    "magical mirai",
    "miku expo",
    "off vocal",
    "premium",
    "project diva",
    "short version",
    "translation",
    "vocal:",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def _tag_info(raw_tag: Any) -> dict[str, Any] | None:
    if isinstance(raw_tag, str):
        return {
            "name": raw_tag,
            "category": None,
            "url_slug": None,
            "usage_count": None,
        }
    if not isinstance(raw_tag, dict):
        return None

    tag = raw_tag.get("tag") or raw_tag.get("Tag") or raw_tag
    if isinstance(tag, dict):
        name = tag.get("name") or tag.get("Name")
        category = tag.get("categoryName") or tag.get("CategoryName")
        url_slug = tag.get("urlSlug") or tag.get("UrlSlug")
    else:
        name = raw_tag.get("name") or raw_tag.get("Name")
        category = raw_tag.get("categoryName") or raw_tag.get("CategoryName")
        url_slug = raw_tag.get("urlSlug") or raw_tag.get("UrlSlug")
    if not name:
        return None
    return {
        "name": name,
        "category": category,
        "url_slug": url_slug,
        "usage_count": raw_tag.get("count") or raw_tag.get("Count"),
    }


def _song_has_artist_role(song: dict[str, Any], artist_id: int) -> bool:
    artists = song.get("artists") or []
    for artist_ref in artists:
        artist = artist_ref.get("artist") if isinstance(artist_ref, dict) else None
        if not isinstance(artist, dict) or artist.get("id") != artist_id:
            continue
        role_text = " ".join(
            str(artist_ref.get(key, ""))
            for key in ("roles", "effectiveRoles", "categories")
        ).lower()
        if any(role in role_text for role in ("composer", "default", "producer")):
            return True
    return False


def _is_original_candidate(song: dict[str, Any], artist_id: int) -> bool:
    song_type = str(song.get("songType", "")).lower()
    if song_type and song_type not in {"original", "remaster", "instrumental"}:
        return False
    return _song_has_artist_role(song, artist_id)


def _is_style_tag(tag: str, category: str | None = None) -> bool:
    normalized = _normalize_tag(tag)
    if not normalized or normalized in NON_STYLE_EXACT:
        return False
    if category in NON_STYLE_CATEGORIES:
        return False
    if category and category not in STYLE_CATEGORY_PRIORITY:
        return False
    return not any(part in normalized for part in NON_STYLE_SUBSTRINGS)


def _normalize_tag(tag: str) -> str:
    return " ".join(tag.strip().casefold().split())


def _song_title(song: dict[str, Any]) -> str:
    return song.get("defaultName") or song.get("name") or f"song {song.get('id')}"


def _aggregate_style_tags(
    songs: Iterable[dict[str, Any]],
    artist_id: int,
    evidence_per_tag: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    aggregates: dict[str, dict[str, Any]] = {}
    analyzed_songs = 0
    for song in songs:
        if not _is_original_candidate(song, artist_id):
            continue
        analyzed_songs += 1
        seen_for_song = set()
        for raw_tag in song.get("tags") or song.get("tagUsages") or []:
            info = _tag_info(raw_tag)
            if not info:
                continue
            normalized = _normalize_tag(info["name"])
            category = info.get("category")
            if normalized in seen_for_song or not _is_style_tag(normalized, category):
                continue
            seen_for_song.add(normalized)
            aggregate = aggregates.setdefault(
                normalized,
                {
                    "label": normalized,
                    "category": category,
                    "song_count": 0,
                    "evidence": [],
                },
            )
            aggregate["song_count"] += 1
            if len(aggregate["evidence"]) < evidence_per_tag:
                song_id = song.get("id")
                aggregate["evidence"].append({
                    "song_id": song_id,
                    "title": _song_title(song),
                    "url": f"https://vocadb.net/S/{song_id}" if song_id else None,
                    "tag_votes": info.get("usage_count"),
                })
    return aggregates, analyzed_songs


def _style_tags_from_aggregates(
    aggregates: dict[str, dict[str, Any]],
    top_tags: int,
) -> list[dict[str, Any]]:
    tags = []
    ranked = sorted(
        aggregates.values(),
        key=lambda item: (
            STYLE_CATEGORY_PRIORITY.get(item.get("category"), 99),
            -item["song_count"],
            item["label"],
        ),
    )
    for item in ranked[:top_tags]:
        tag = item["label"]
        tags.append({
            "label": tag,
            "display_zh": TAG_TRANSLATIONS.get(tag, tag),
            "category": item.get("category"),
            "song_count": item["song_count"],
            "evidence": item["evidence"],
        })
    return tags


def _fetch_songs(client: VocaDBClient, artist_id: int, max_songs: int) -> list[dict[str, Any]]:
    return client.get_songs_by_artist(
        artist_id,
        fields="PVs,Artists,Tags",
        max_results=max_songs,
        sort="RatingScore",
    )


def build_style_config(args: argparse.Namespace) -> dict[str, Any]:
    producers_config = _load_yaml(args.producers)
    existing_config = _load_yaml(args.output)
    existing_by_slug = existing_config.get("producers", {})
    client = None if args.no_api else VocaDBClient(user_agent=args.user_agent)

    output: dict[str, Any] = {
        "source": {
            "name": "VocaDB song tags",
            "url": "https://vocadb.net/",
            "api_url": "https://vocadb.net/api",
            "refresh_script": "scripts/22_refresh_vocadb_style_tags.py",
            "last_reviewed": args.reviewed_date,
            "notes": [
                "Runtime never calls VocaDB; tags are cached in this file for stable deployment.",
                "Tags are display-only descriptors and are not used by the audio model.",
                "Songs are fetched with the VocaDB artistId[] query parameter and ranked by RatingScore.",
                "Re-run the refresh script from an environment that can access VocaDB or from browser-exported raw JSONL.",
            ],
        },
        "producers": {},
    }

    for producer in producers_config.get("producers", []):
        slug = producer["slug"]
        artist_id = int(producer["vocadb_artist_id"])
        songs: list[dict[str, Any]] = []

        raw_path = args.raw_dir / f"{slug}_songs.jsonl"
        if client is not None:
            try:
                songs = _fetch_songs(client, artist_id, args.max_songs)
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[warn] {slug}: VocaDB fetch failed: {exc}")

        if not songs:
            songs.extend(_load_jsonl(raw_path))

        aggregates, songs_analyzed = _aggregate_style_tags(
            songs,
            artist_id,
            args.evidence_per_tag,
        )
        tags = _style_tags_from_aggregates(aggregates, args.top_tags)

        if not tags:
            tags = existing_by_slug.get(slug, {}).get("style_tags", [])
            if tags:
                print(f"[keep] {slug}: no usable VocaDB rows; kept existing tags")
            else:
                print(f"[warn] {slug}: no style tags found")
        else:
            print(
                f"[ok] {slug}: "
                f"{', '.join(f'{tag['label']}({tag['song_count']})' for tag in tags)}"
            )

        output["producers"][slug] = {
            "artist_id": artist_id,
            "source_url": producer.get("profile_url") or f"https://vocadb.net/Ar/{artist_id}",
            "api_url": f"https://vocadb.net/api/songs?artistId[]={artist_id}&fields=PVs,Artists,Tags",
            "songs_analyzed": songs_analyzed,
            "style_tags": tags,
        }

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producers", type=Path, default=Path("configs/producers.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_jsonl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-songs", type=int, default=200)
    parser.add_argument("--top-tags", type=int, default=3)
    parser.add_argument("--evidence-per-tag", type=int, default=3)
    parser.add_argument("--reviewed-date", default="2026-06-26")
    parser.add_argument("--user-agent", default="vocaptest/0.1 style-tag-refresh")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Only read raw JSONL files; do not attempt live VocaDB requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_style_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
