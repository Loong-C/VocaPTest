#!/usr/bin/env python
"""Prepare a vetted top100 producer expansion batch.

This script intentionally stops at catalog/config generation.  Audio download,
VocaDB PV validation, embedding rebuilds, and model training stay in the
existing reproducible scripts.

Selection policy:
- use VocaDB rating order as the first popularity signal;
- require Original songs, voice-synth credits, and target producer Composer or
  Default roles;
- accept enabled VocaDB Original PVs from Youtube or NicoNicoDouga;
- skip Topic/SEGA/Project Sekai/KARENT-style uploads and unapproved external
  style-credit collaborators;
- allow partial train/dev/final splits for genuinely sparse producers instead
  of skipping the producer solely because the ideal split cannot be filled.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
DEFAULT_BATCH = (
    "kuro_usa_p",
    "mothy",
    "hiiragi_magnetite",
    "owata_p",
    "nuyuri",
)


@dataclass(frozen=True)
class Candidate:
    rank: int
    slug: str
    display_name: str
    aliases: tuple[str, ...]
    vocadb_artist_id: int
    top100_name: str
    allowed_style_credit_ids: tuple[int, ...] = ()
    reason: str = ""


CANDIDATES: dict[str, Candidate] = {
    "giga": Candidate(
        rank=14,
        slug="giga",
        display_name="Giga",
        aliases=("ギガ",),
        vocadb_artist_id=772,
        top100_name="ギガ",
    ),
    "rerulili": Candidate(
        rank=15,
        slug="rerulili",
        display_name="れるりり",
        aliases=("rerulili", "当社比P", "ToushahiP"),
        vocadb_artist_id=712,
        top100_name="れるりり",
    ),
    "ryo": Candidate(
        rank=19,
        slug="ryo",
        display_name="ryo",
        aliases=("supercell",),
        vocadb_artist_id=67,
        top100_name="ryo",
        allowed_style_credit_ids=(249,),
        reason="Allow supercell as ryo's own project credit.",
    ),
    "mikito_p": Candidate(
        rank=20,
        slug="mikito_p",
        display_name="みきとP",
        aliases=("MikitoP", "愛島", "Aijima"),
        vocadb_artist_id=876,
        top100_name="みきとP",
    ),
    "hitoshizuku_p": Candidate(
        rank=22,
        slug="hitoshizuku_p",
        display_name="ひとしずくP / やま△",
        aliases=("ひとしずくP", "HitoshizukuP", "やま△", "Yama△", "さも", "samo"),
        vocadb_artist_id=103,
        top100_name="ひとしずくP",
        allowed_style_credit_ids=(499,),
    ),
    "balloon": Candidate(
        rank=23,
        slug="balloon",
        display_name="バルーン",
        aliases=("balloon", "須田景凪", "Suda Keina"),
        vocadb_artist_id=10259,
        top100_name="バルーン",
    ),
    "kuro_usa_p": Candidate(
        rank=24,
        slug="kuro_usa_p",
        display_name="黒うさP",
        aliases=("KurousaP", "くろうさP", "WhiteFlame", "しゃな", "syana"),
        vocadb_artist_id=310,
        top100_name="黒うさ",
    ),
    "mothy": Candidate(
        rank=25,
        slug="mothy",
        display_name="mothy",
        aliases=("悪ノP", "AkunoP", "master of the heavenly yard"),
        vocadb_artist_id=189,
        top100_name="mothy",
    ),
    "hiiragi_magnetite": Candidate(
        rank=27,
        slug="hiiragi_magnetite",
        display_name="柊マグネタイト",
        aliases=("Hiiragi Magnetite",),
        vocadb_artist_id=83243,
        top100_name="柊マグネタイト",
    ),
    "owata_p": Candidate(
        rank=28,
        slug="owata_p",
        display_name="オワタP",
        aliases=("OwataP", "ガルナ", "Garuna"),
        vocadb_artist_id=94,
        top100_name="ガルナ@オワタP",
    ),
    "nuyuri": Candidate(
        rank=29,
        slug="nuyuri",
        display_name="ぬゆり",
        aliases=("Nuyuri", "nulut", "Lanndo", "go乱心P", "ぬるり", "Crona"),
        vocadb_artist_id=5666,
        top100_name="ぬゆり",
    ),
    "eve": Candidate(
        rank=39,
        slug="eve",
        display_name="Eve",
        aliases=(),
        vocadb_artist_id=10233,
        top100_name="Eve",
    ),
    "papiyon": Candidate(
        rank=43,
        slug="papiyon",
        display_name="蝶々P",
        aliases=("papiyon", "一之瀬ユウ", "Yu Ichinose"),
        vocadb_artist_id=96,
        top100_name="papiyon/蝶々P",
    ),
    "wotaku": Candidate(
        rank=45,
        slug="wotaku",
        display_name="wotaku",
        aliases=(),
        vocadb_artist_id=60331,
        top100_name="wotaku",
    ),
    "ume_tora": Candidate(
        rank=46,
        slug="ume_tora",
        display_name="梅とら",
        aliases=("umedy",),
        vocadb_artist_id=1164,
        top100_name="梅とら",
    ),
    "hachioji_p": Candidate(
        rank=47,
        slug="hachioji_p",
        display_name="八王子P",
        aliases=("HachiojiP", "8#Prince"),
        vocadb_artist_id=38,
        top100_name="八王子P",
    ),
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


def read_top100(path: Path) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^(\d+)\s*[\uff0c,、]\s*(.+)$", line.strip())
        if match:
            ranks[match.group(2).strip()] = int(match.group(1))
    return ranks


def split_roles(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


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
        time.sleep(0.3)
    return songs


def has_voice_synth(song: dict) -> bool:
    for credit in song.get("artists", []):
        artist = credit.get("artist") or {}
        if artist.get("artistType") in VOICE_SYNTH_TYPES:
            return True
    return False


def style_roles_for(song: dict, artist_id: int) -> set[str]:
    roles: set[str] = set()
    for credit in song.get("artists", []):
        if (credit.get("artist") or {}).get("id") == artist_id:
            roles.update(split_roles(credit.get("effectiveRoles", "")))
    return roles


def original_youtube_pvs(song: dict) -> list[dict]:
    return [
        pv
        for pv in song.get("pvs", [])
        if pv.get("service") == "Youtube"
        and pv.get("pvType") == "Original"
        and pv.get("pvId")
        and not pv.get("disabled")
    ]


def original_media_pvs(song: dict) -> list[dict]:
    return [
        pv
        for pv in song.get("pvs", [])
        if pv.get("service") in {"Youtube", "NicoNicoDouga"}
        and pv.get("pvType") == "Original"
        and pv.get("pvId")
        and not pv.get("disabled")
    ]


def is_risky_pv_author(author: str | None) -> bool:
    folded = (author or "").casefold()
    return any(term in folded for term in RISKY_PV_AUTHOR_TERMS)


def preferred_original_media_pvs(song: dict) -> list[dict]:
    service_order = {"Youtube": 0, "NicoNicoDouga": 1}
    pvs = [
        pv
        for pv in original_media_pvs(song)
        if not is_risky_pv_author(pv.get("author"))
    ]
    return sorted(
        pvs,
        key=lambda pv: (
            service_order.get(str(pv.get("service")), 99),
            str(pv.get("pvId")),
        ),
    )


def source_for_entry(item: dict) -> tuple[str, str] | None:
    if item.get("source_service") and item.get("source_id"):
        return str(item["source_service"]), str(item["source_id"])
    if item.get("youtube_id"):
        return "Youtube", str(item["youtube_id"])
    return None


def existing_ids(root: Path) -> tuple[set[int], set[tuple[str, str]]]:
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
    vocadb_ids: set[int] = set()
    source_ids: set[tuple[str, str]] = set()
    for item in [song for path in yaml_paths for song in load_yaml_songs(path)]:
        if item.get("vocadb_song_id") is not None:
            vocadb_ids.add(int(item["vocadb_song_id"]))
        source = source_for_entry(item)
        if source:
            source_ids.add(source)
    for item in [record for path in jsonl_paths for record in load_jsonl(path)]:
        if item.get("vocadb_song_id") is not None:
            vocadb_ids.add(int(item["vocadb_song_id"]))
        if item.get("source_service") and item.get("source_id"):
            source_ids.add((str(item["source_service"]), str(item["source_id"])))
        else:
            song_id = str(item.get("song_id", ""))
            if song_id.startswith("youtube_"):
                source_ids.add(("Youtube", song_id.removeprefix("youtube_")))
            elif song_id.startswith("niconico_"):
                source_ids.add(("NicoNicoDouga", song_id.removeprefix("niconico_")))
    return vocadb_ids, source_ids


def configured_artist_ids(root: Path) -> dict[str, int]:
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        producers = (yaml.safe_load(handle) or {}).get("producers", [])
    return {
        item["slug"]: int(item["vocadb_artist_id"])
        for item in producers
        if item.get("vocadb_artist_id") is not None
    }


def has_excluded_style_credit(song: dict, excluded_artist_ids: set[int]) -> bool:
    for credit in song.get("artists", []):
        artist_id = (credit.get("artist") or {}).get("id")
        if artist_id not in excluded_artist_ids:
            continue
        if split_roles(credit.get("effectiveRoles", "")) & STYLE_ROLES:
            return True
    return False


def has_unapproved_external_style_credit(song: dict, candidate: Candidate) -> bool:
    allowed = {candidate.vocadb_artist_id, *candidate.allowed_style_credit_ids}
    for credit in song.get("artists", []):
        artist = credit.get("artist") or {}
        artist_id = artist.get("id")
        if artist_id in allowed:
            continue
        artist_type = artist.get("artistType")
        if artist_type not in {"OtherGroup", "Producer"}:
            continue
        if split_roles(credit.get("effectiveRoles", "")) & STYLE_ROLES:
            return True
    return False


def select_catalog_entries(
    session: requests.Session,
    candidate: Candidate,
    *,
    existing_vocadb_ids: set[int],
    existing_source_ids: set[tuple[str, str]],
    excluded_artist_ids: set[int],
    max_results: int,
) -> list[dict]:
    selected: list[dict] = []
    seen_vocadb = set(existing_vocadb_ids)
    seen_sources = set(existing_source_ids)
    for song in fetch_songs(session, candidate.vocadb_artist_id, max_results):
        roles = style_roles_for(song, candidate.vocadb_artist_id)
        if not (roles & STYLE_ROLES):
            continue
        if not has_voice_synth(song):
            continue
        if has_excluded_style_credit(song, excluded_artist_ids):
            continue
        if has_unapproved_external_style_credit(song, candidate):
            continue
        if int(song["id"]) in seen_vocadb:
            continue
        for pv in preferred_original_media_pvs(song):
            source_service = str(pv["service"])
            source_id = str(pv["pvId"])
            source = (source_service, source_id)
            if source in seen_sources:
                continue
            entry = {
                "producer_slug": candidate.slug,
                "title": song.get("name") or song.get("defaultName") or str(song["id"]),
                "vocadb_song_id": int(song["id"]),
                "source_service": source_service,
                "source_id": source_id,
                "source_kind": "vocadb_original_pv",
            }
            if source_service == "Youtube":
                entry["youtube_id"] = source_id
            selected.append(entry)
            seen_vocadb.add(int(song["id"]))
            seen_sources.add(source)
            break
        if len(selected) >= max_results:
            return selected
    return selected


def split_selected(
    selected: list[dict],
    *,
    train_count: int,
    dev_count: int,
    final_count: int,
    min_train_count: int,
    min_dev_count: int,
    min_final_count: int,
) -> dict[str, list[dict]] | None:
    total = len(selected)
    if total < min_train_count + min_dev_count + min_final_count:
        return None
    train_n = min(train_count, total - min_dev_count - min_final_count)
    dev_n = min(dev_count, total - train_n - min_final_count)
    final_n = min(final_count, total - train_n - dev_n)
    if (
        train_n < min_train_count
        or dev_n < min_dev_count
        or final_n < min_final_count
    ):
        return None
    return {
        "train": selected[:train_n],
        "dev": selected[train_n: train_n + dev_n],
        "final": selected[train_n + dev_n: train_n + dev_n + final_n],
    }


def producer_exists(root: Path, slug: str) -> bool:
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        producers = (yaml.safe_load(handle) or {}).get("producers", [])
    return any(item.get("slug") == slug for item in producers)


def format_scalar(value: object) -> str:
    text = str(value)
    if not text or text.startswith((" ", "-", "{", "[")) or ":" in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def producer_block(candidate: Candidate) -> str:
    lines = [
        f"- slug: {candidate.slug}",
        f"  display_name: {candidate.display_name}",
    ]
    if candidate.aliases:
        lines.append("  aliases:")
        lines.extend(f"  - {alias}" for alias in candidate.aliases)
    else:
        lines.append("  aliases: []")
    lines.extend([
        f"  vocadb_artist_id: {candidate.vocadb_artist_id}",
        f"  profile_url: https://vocadb.net/Ar/{candidate.vocadb_artist_id}",
        f"  avatar_url: /avatars/{candidate.slug}.webp",
    ])
    return "\n".join(lines) + "\n"


def catalog_block(entries: Iterable[dict], indent: str) -> str:
    blocks: list[str] = []
    for entry in entries:
        lines = [
            f"{indent}- producer_slug: {entry['producer_slug']}",
            f"{indent}  title: {format_scalar(entry['title'])}",
            f"{indent}  vocadb_song_id: {entry['vocadb_song_id']}",
            f"{indent}  source_service: {entry['source_service']}",
            f"{indent}  source_id: {entry['source_id']}",
            f"{indent}  source_kind: {entry['source_kind']}",
        ]
        if entry.get("youtube_id"):
            lines.insert(4, f"{indent}  youtube_id: {entry['youtube_id']}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


def list_indent(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("- "):
            return line[: len(line) - len(line.lstrip())]
    return "  "


def append_text(path: Path, text: str) -> None:
    existing = path.read_text(encoding="utf-8")
    separator = "" if existing.endswith("\n") else "\n"
    path.write_text(existing + separator + text, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top100", type=Path, default=root / "data" / "raw" / "top100p.txt")
    parser.add_argument("--slug", action="append", default=[])
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--dev-count", type=int, default=2)
    parser.add_argument("--final-count", type=int, default=4)
    parser.add_argument("--min-train-count", type=int, default=2)
    parser.add_argument("--min-dev-count", type=int, default=0)
    parser.add_argument("--min-final-count", type=int, default=1)
    parser.add_argument("--max-results", type=int, default=160)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested = tuple(args.slug or DEFAULT_BATCH)
    unknown = sorted(set(requested).difference(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown expansion candidate(s): {unknown}")

    top100 = read_top100(args.top100)
    session = requests.Session()
    session.headers["User-Agent"] = "VocaPTest/0.1 top100 expansion"
    existing_vocadb_ids, existing_source_ids = existing_ids(root)
    current_artist_ids = configured_artist_ids(root)

    prepared: dict[str, dict[str, list[dict]]] = {}
    skipped: dict[str, str] = {}
    desired = args.train_count + args.dev_count + args.final_count
    for slug in requested:
        candidate = CANDIDATES[slug]
        if candidate.top100_name not in top100:
            skipped[slug] = f"{candidate.top100_name} not present in top100 file"
            continue
        selected = select_catalog_entries(
            session,
            candidate,
            existing_vocadb_ids=existing_vocadb_ids,
            existing_source_ids=existing_source_ids,
            excluded_artist_ids={
                artist_id
                for slug, artist_id in current_artist_ids.items()
                if artist_id != candidate.vocadb_artist_id
            },
            max_results=args.max_results,
        )
        splits = split_selected(
            selected,
            train_count=args.train_count,
            dev_count=args.dev_count,
            final_count=args.final_count,
            min_train_count=args.min_train_count,
            min_dev_count=args.min_dev_count,
            min_final_count=args.min_final_count,
        )
        if splits is None:
            skipped[slug] = (
                f"only {len(selected)} verified voice-synth Original PVs; "
                "not enough for the minimum train/dev/final split"
            )
            continue
        prepared[slug] = splits
        used = [item for split_items in splits.values() for item in split_items]
        if len(used) < desired:
            skipped[f"{slug}_partial_holdout"] = (
                f"using {len(used)} songs instead of desired {desired}; "
                f"split={ {key: len(value) for key, value in splits.items()} }"
            )
        existing_vocadb_ids.update(item["vocadb_song_id"] for item in used)
        existing_source_ids.update(
            source
            for item in used
            for source in [source_for_entry(item)]
            if source
        )

    report = {
        "requested": list(requested),
        "prepared": {
            slug: {split: len(items) for split, items in splits.items()}
            for slug, splits in prepared.items()
        },
        "skipped": skipped,
        "songs": prepared,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    producers_path = root / "configs" / "producers.yaml"
    for slug in prepared:
        if not producer_exists(root, slug):
            append_text(producers_path, producer_block(CANDIDATES[slug]))

    catalog_paths = {
        "train": root / "configs" / "training_catalog_additions.yaml",
        "dev": root / "configs" / "dev_holdout_catalog.yaml",
        "final": root / "configs" / "frozen_test_catalog.yaml",
    }
    for split, path in catalog_paths.items():
        indent = list_indent(path)
        entries = [
            entry
            for slug in prepared
            for entry in prepared[slug][split]
        ]
        if entries:
            append_text(path, catalog_block(entries, indent))


if __name__ == "__main__":
    main()
