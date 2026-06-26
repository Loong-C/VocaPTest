#!/usr/bin/env python
"""Audit configured producers for high-rating VocaDB Original PV gaps."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocaptest.data.catalog_sources import media_source_from_item
from vocaptest.utils.paths import project_root


VOCADB_API = "https://vocadb.net/api"
VOICE_SYNTH_TYPES = {
    "CeVIO",
    "NewType",
    "OtherVoiceSynthesizer",
    "SynthesizerV",
    "UTAU",
    "Vocaloid",
    "Voiceroid",
}
STYLE_ROLES = {"Composer", "Default"}
PRODUCER_TYPES = {"OtherGroup", "Producer"}
ALLOWED_STYLE_COLLABORATORS = {
    "hitoshizuku_p": {499},
}
RISKY_PV_AUTHOR_TERMS = (
    "hatsune miku",
    "hatsunemiku",
    "karent",
    "project sekai",
    "sega",
    "topic",
    "vocaloid",
    "プロジェクトセカイ",
    "初音ミク",
)


def load_yaml_songs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("songs", [])


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_roles(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def source_for_record(item: dict) -> tuple[str | None, str | None]:
    try:
        return media_source_from_item(item)
    except ValueError:
        song_id = str(item.get("song_id", ""))
        if song_id.startswith("youtube_"):
            return "Youtube", song_id.removeprefix("youtube_")
        if song_id.startswith("niconico_"):
            return "NicoNicoDouga", song_id.removeprefix("niconico_")
    return None, None


def load_producers(root: Path) -> list[dict]:
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("producers", [])


def configured_catalog(root: Path) -> tuple[dict[str, set[int]], dict[str, set[tuple[str, str]]]]:
    yaml_paths = [
        root / "configs" / "training_catalog_additions.yaml",
        root / "configs" / "dev_holdout_catalog.yaml",
        root / "configs" / "frozen_test_catalog.yaml",
    ]
    jsonl_paths = [
        root / "data" / "processed" / "curated" / "mert_95" / "song_decisions.jsonl",
        root / "data" / "processed" / "dev_holdout" / "catalog.jsonl",
        root / "data" / "processed" / "frozen_test" / "catalog.jsonl",
    ]
    vocadb_ids: dict[str, set[int]] = defaultdict(set)
    source_ids: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for item in [song for path in yaml_paths for song in load_yaml_songs(path)]:
        slug = str(item.get("producer_slug"))
        if item.get("vocadb_song_id") is not None:
            vocadb_ids[slug].add(int(item["vocadb_song_id"]))
        source = source_for_record(item)
        if all(source):
            source_ids[slug].add((str(source[0]), str(source[1])))
    for item in [record for path in jsonl_paths for record in load_jsonl(path)]:
        if item.get("status") not in {None, "accepted"}:
            continue
        slug = str(item.get("producer_slug"))
        if item.get("vocadb_song_id") is not None:
            vocadb_ids[slug].add(int(item["vocadb_song_id"]))
        source = source_for_record(item)
        if all(source):
            source_ids[slug].add((str(source[0]), str(source[1])))
    return vocadb_ids, source_ids


def fetch_songs(session: requests.Session, artist_id: int, max_results: int) -> list[dict]:
    songs: list[dict] = []
    start = 0
    while len(songs) < max_results:
        response = session.get(
            f"{VOCADB_API}/songs",
            params={
                "artistId[]": artist_id,
                "fields": "Artists,PVs",
                "maxResults": min(50, max_results - len(songs)),
                "songTypes": "Original",
                "sort": "RatingScore",
                "start": start,
            },
            timeout=60,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        songs.extend(items)
        if len(items) < 50:
            break
        start += len(items)
        time.sleep(0.2)
    return songs


def has_voice_synth(song: dict) -> bool:
    return any(
        (credit.get("artist") or {}).get("artistType") in VOICE_SYNTH_TYPES
        for credit in song.get("artists", [])
    )


def style_roles_for(song: dict, artist_id: int) -> set[str]:
    roles: set[str] = set()
    for credit in song.get("artists", []):
        if (credit.get("artist") or {}).get("id") == artist_id:
            roles.update(split_roles(credit.get("effectiveRoles", "")))
    return roles


def has_unapproved_external_style_credit(
    song: dict,
    *,
    slug: str,
    artist_id: int,
    configured_artist_ids: set[int],
) -> bool:
    allowed = {artist_id, *ALLOWED_STYLE_COLLABORATORS.get(slug, set())}
    for credit in song.get("artists", []):
        artist = credit.get("artist") or {}
        other_id = artist.get("id")
        if other_id in allowed:
            continue
        if artist.get("artistType") not in PRODUCER_TYPES:
            continue
        if not (split_roles(credit.get("effectiveRoles", "")) & STYLE_ROLES):
            continue
        if other_id in configured_artist_ids or artist.get("artistType") in PRODUCER_TYPES:
            return True
    return False


def is_risky_pv_author(author: str | None) -> bool:
    folded = (author or "").casefold()
    return any(term in folded for term in RISKY_PV_AUTHOR_TERMS)


def original_media_pvs(song: dict) -> list[dict]:
    return [
        pv
        for pv in song.get("pvs", [])
        if pv.get("service") in {"Youtube", "NicoNicoDouga"}
        and pv.get("pvType") == "Original"
        and pv.get("pvId")
        and not pv.get("disabled")
        and not is_risky_pv_author(pv.get("author"))
    ]


def best_pv(song: dict) -> tuple[str, str] | None:
    service_order = {"Youtube": 0, "NicoNicoDouga": 1}
    pvs = sorted(
        original_media_pvs(song),
        key=lambda pv: (
            service_order.get(str(pv.get("service")), 99),
            str(pv.get("pvId")),
        ),
    )
    if not pvs:
        return None
    return str(pvs[0]["service"]), str(pvs[0]["pvId"])


def is_configured(
    song: dict,
    *,
    slug: str,
    configured_vocadb_ids: dict[str, set[int]],
    configured_source_ids: dict[str, set[tuple[str, str]]],
) -> bool:
    if int(song["id"]) in configured_vocadb_ids.get(slug, set()):
        return True
    song_sources = {
        (str(pv["service"]), str(pv["pvId"]))
        for pv in original_media_pvs(song)
    }
    return bool(song_sources & configured_source_ids.get(slug, set()))


def audit_producer(
    session: requests.Session,
    producer: dict,
    *,
    configured_vocadb_ids: dict[str, set[int]],
    configured_source_ids: dict[str, set[tuple[str, str]]],
    configured_artist_ids: set[int],
    max_results: int,
    top_missing: int,
) -> dict:
    slug = str(producer["slug"])
    artist_id = int(producer["vocadb_artist_id"])
    qualified = []
    missing = []
    for song in fetch_songs(session, artist_id, max_results):
        if not (style_roles_for(song, artist_id) & STYLE_ROLES):
            continue
        if not has_voice_synth(song):
            continue
        if has_unapproved_external_style_credit(
            song,
            slug=slug,
            artist_id=artist_id,
            configured_artist_ids=configured_artist_ids,
        ):
            continue
        source = best_pv(song)
        if source is None:
            continue
        item = {
            "vocadb_song_id": int(song["id"]),
            "title": song.get("name") or song.get("defaultName") or str(song["id"]),
            "rating_score": song.get("ratingScore") or 0,
            "source_service": source[0],
            "source_id": source[1],
        }
        qualified.append(item)
        if not is_configured(
            song,
            slug=slug,
            configured_vocadb_ids=configured_vocadb_ids,
            configured_source_ids=configured_source_ids,
        ):
            missing.append(item)
    configured_count = max(
        len(configured_vocadb_ids.get(slug, set())),
        len(configured_source_ids.get(slug, set())),
    )
    return {
        "slug": slug,
        "display_name": producer.get("display_name"),
        "vocadb_artist_id": artist_id,
        "configured_vocadb_song_count": len(configured_vocadb_ids.get(slug, set())),
        "configured_source_count": len(configured_source_ids.get(slug, set())),
        "configured_count_lower_bound": configured_count,
        "qualified_available_count": len(qualified),
        "unconfigured_qualified_count": len(missing),
        "top_unconfigured": missing[:top_missing],
    }


def write_report(path: Path, audit: dict) -> None:
    lines = [
        "# VocaDB Catalog Coverage Audit",
        "",
        "This report lists high-rating VocaDB Original PVs that are eligible by the current catalog rules but are not present in the configured train/dev/final splits.",
        "",
        "A listed song is a review candidate, not an automatic addition.",
        "",
        "## Summary",
        "",
        f"- Producers audited: {audit['summary']['producers']}",
        f"- Producers with review candidates: {audit['summary']['producers_with_candidates']}",
        "",
        "## Review Candidates",
        "",
    ]
    for item in audit["producers"]:
        if not item["top_unconfigured"]:
            continue
        lines.extend([
            f"### {item['slug']}",
            "",
            (
                f"Configured source count: {item['configured_source_count']}; "
                f"qualified VocaDB Original PVs found: {item['qualified_available_count']}."
            ),
            "",
            "| Rating | VocaDB | Title | Source |",
            "|---:|---:|---|---|",
        ])
        for song in item["top_unconfigured"]:
            title = str(song["title"]).replace("|", "\\|")
            source = f"{song['source_service']}:{song['source_id']}"
            lines.append(
                f"| {song['rating_score']} | {song['vocadb_song_id']} | {title} | `{source}` |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "evaluations" / "vocadb_catalog_coverage_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs" / "VOCADB_CATALOG_COVERAGE_AUDIT.md",
    )
    parser.add_argument("--max-results", type=int, default=120)
    parser.add_argument("--top-missing", type=int, default=6)
    args = parser.parse_args()

    producers = [
        item for item in load_producers(root)
        if item.get("vocadb_artist_id") is not None
    ]
    configured_vocadb_ids, configured_source_ids = configured_catalog(root)
    configured_artist_ids = {
        int(item["vocadb_artist_id"])
        for item in producers
        if item.get("vocadb_artist_id") is not None
    }
    session = requests.Session()
    session.headers["User-Agent"] = "VocaPTest/0.1 catalog coverage audit"
    audited = [
        audit_producer(
            session,
            producer,
            configured_vocadb_ids=configured_vocadb_ids,
            configured_source_ids=configured_source_ids,
            configured_artist_ids=configured_artist_ids,
            max_results=args.max_results,
            top_missing=args.top_missing,
        )
        for producer in producers
    ]
    audit = {
        "protocol": {
            "source": VOCADB_API,
            "purpose": "Find eligible, high-rating Original PV catalog gaps before producer expansion.",
        },
        "summary": {
            "producers": len(audited),
            "producers_with_candidates": sum(bool(item["top_unconfigured"]) for item in audited),
        },
        "producers": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report_output, audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
