#!/usr/bin/env python
"""Audit training/dev/final catalogs for style-label risk factors."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

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
PRODUCER_TYPES = {"OtherGroup", "Producer"}
ALLOWED_STYLE_COLLABORATORS = {
    "hitoshizuku_p": {499},  # やま△, a stable part of the catalogued duo style.
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


def source_for_record(item: dict) -> tuple[str | None, str | None]:
    if item.get("source_service") and item.get("source_id"):
        return str(item["source_service"]), str(item["source_id"])
    if item.get("youtube_id"):
        return "Youtube", str(item["youtube_id"])
    song_id = str(item.get("song_id", ""))
    if song_id.startswith("youtube_"):
        return "Youtube", song_id.removeprefix("youtube_")
    if song_id.startswith("niconico_"):
        return "NicoNicoDouga", song_id.removeprefix("niconico_")
    return None, None


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


def producer_map(root: Path) -> dict[str, dict]:
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        return {
            item["slug"]: item
            for item in (yaml.safe_load(handle) or {}).get("producers", [])
        }


def split_roles(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def collect_records(root: Path) -> list[dict]:
    sources = [
        ("train", root / "data" / "processed" / "curated" / "mert_95" / "song_decisions.jsonl", "jsonl"),
        ("dev", root / "data" / "processed" / "dev_holdout" / "catalog.jsonl", "jsonl"),
        ("final", root / "data" / "processed" / "frozen_test" / "catalog.jsonl", "jsonl"),
        ("train", root / "configs" / "training_catalog_additions.yaml", "yaml"),
        ("dev", root / "configs" / "dev_holdout_catalog.yaml", "yaml"),
        ("final", root / "configs" / "frozen_test_catalog.yaml", "yaml"),
    ]
    records: dict[tuple[str, str, int | None, str | None], dict] = {}
    for split, path, kind in sources:
        items = load_jsonl(path) if kind == "jsonl" else load_yaml_songs(path)
        for item in items:
            if kind == "jsonl" and item.get("status") not in {None, "accepted"}:
                continue
            source_service, source_id = source_for_record(item)
            vocadb_id = item.get("vocadb_song_id")
            key = (
                split,
                str(item.get("producer_slug")),
                int(vocadb_id) if vocadb_id is not None else None,
                source_service,
                source_id,
            )
            existing = records.setdefault(key, {
                "split": split,
                "producer_slug": item.get("producer_slug"),
                "title": item.get("title"),
                "vocadb_song_id": int(vocadb_id) if vocadb_id is not None else None,
                "source_service": source_service,
                "source_id": source_id,
                "youtube_id": source_id if source_service == "Youtube" else None,
                "source_kind": item.get("source_kind"),
                "sources": [],
            })
            existing["sources"].append(path.as_posix())
    return sorted(
        records.values(),
        key=lambda item: (
            item["split"],
            str(item["producer_slug"]),
            str(item.get("title")),
        ),
    )


def fetch_song(
    session: requests.Session,
    song_id: int,
    cache_dir: Path,
    use_cache: bool,
) -> dict:
    cache_path = cache_dir / f"{song_id}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = session.get(
        f"{VOCADB_API}/songs/{song_id}",
        params={"fields": "Artists,PVs,Tags"},
        timeout=60,
    )
    response.raise_for_status()
    song = response.json()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(song, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.15)
    return song


def is_risky_pv_author(author: str | None) -> bool:
    folded = (author or "").casefold()
    return any(term in folded for term in RISKY_PV_AUTHOR_TERMS)


def analyze_record(record: dict, song: dict | None, producers: dict[str, dict]) -> dict:
    flags: list[dict] = []

    def flag(code: str, severity: str, detail: str) -> None:
        flags.append({"code": code, "severity": severity, "detail": detail})

    slug = str(record.get("producer_slug"))
    producer = producers.get(slug)
    expected_artist_id = (
        int(producer["vocadb_artist_id"])
        if producer and producer.get("vocadb_artist_id") is not None
        else None
    )
    if not producer:
        flag("unknown_producer_slug", "high", f"{slug} is not in configs/producers.yaml")
    if record.get("vocadb_song_id") is None:
        flag("missing_vocadb_song_id", "high", "No VocaDB song id is configured")
    if song is None:
        return {**record, "flags": flags, "risk_score": score_flags(flags)}

    if song.get("songType") != "Original":
        flag("song_type_not_original", "high", f"VocaDB songType={song.get('songType')}")
    if (song.get("ratingScore") or 0) < 20:
        flag("low_vocadb_rating", "low", f"ratingScore={song.get('ratingScore')}")

    voice_synths = []
    expected_roles: set[str] = set()
    other_style_credits = []
    configured_style_credits = []
    allowed_collaborators = ALLOWED_STYLE_COLLABORATORS.get(slug, set())
    configured_artist_ids = {
        str(item["slug"]): int(item["vocadb_artist_id"])
        for item in producers.values()
        if item.get("vocadb_artist_id") is not None
    }
    configured_ids_to_slug = {artist_id: item_slug for item_slug, artist_id in configured_artist_ids.items()}

    for credit in song.get("artists", []):
        artist = credit.get("artist") or {}
        artist_id = artist.get("id")
        roles = split_roles(credit.get("effectiveRoles", ""))
        if artist.get("artistType") in VOICE_SYNTH_TYPES:
            voice_synths.append(artist.get("name"))
        if expected_artist_id is not None and artist_id == expected_artist_id:
            expected_roles.update(roles)
        if (
            artist_id != expected_artist_id
            and artist_id not in allowed_collaborators
            and artist.get("artistType") in PRODUCER_TYPES
            and roles & STYLE_ROLES
        ):
            other_style_credits.append({
                "id": artist_id,
                "name": artist.get("name"),
                "artist_type": artist.get("artistType"),
                "roles": sorted(roles),
            })
            if artist_id in configured_ids_to_slug:
                configured_style_credits.append(configured_ids_to_slug[artist_id])

    if expected_artist_id is not None and not expected_roles:
        flag("expected_artist_missing", "high", f"artist_id={expected_artist_id} has no credit")
    elif not (expected_roles & STYLE_ROLES):
        flag("expected_artist_not_style_credit", "high", f"roles={sorted(expected_roles)}")
    if not voice_synths:
        flag("no_voice_synth_credit", "high", "No Vocaloid/UTAU/CeVIO/SynthV-style singer credit found")
    for other in other_style_credits:
        severity = "medium"
        if other["id"] in configured_ids_to_slug:
            severity = "high"
        flag(
            "external_style_credit",
            severity,
            f"{other['name']} ({other['artist_type']}) roles={','.join(other['roles'])}",
        )
    if configured_style_credits:
        flag(
            "overlaps_configured_producer",
            "high",
            f"also credited to configured slug(s): {sorted(configured_style_credits)}",
        )

    source_service = record.get("source_service")
    source_id = record.get("source_id")
    source_pvs = [
        pv for pv in song.get("pvs", [])
        if pv.get("service") == source_service and pv.get("pvId") == source_id
    ]
    if source_id and not source_pvs:
        flag(
            "configured_source_pv_missing",
            "high",
            f"{source_service} {source_id} not listed on VocaDB song",
        )
    for pv in source_pvs:
        pv_type = pv.get("pvType")
        author = pv.get("author")
        if pv_type != "Original":
            flag("configured_source_not_original", "medium", f"pvType={pv_type}")
        if is_risky_pv_author(author):
            flag("review_pv_author", "low", f"PV author={author}")

    return {
        **record,
        "vocadb_name": song.get("name"),
        "rating_score": song.get("ratingScore"),
        "expected_roles": sorted(expected_roles),
        "voice_synths": voice_synths,
        "flags": flags,
        "risk_score": score_flags(flags),
    }


def score_flags(flags: list[dict]) -> int:
    weights = {"high": 100, "medium": 10, "low": 1}
    return sum(weights.get(flag["severity"], 0) for flag in flags)


def write_report(path: Path, audit: dict) -> None:
    severity_order = {"high": 0, "medium": 1, "low": 2}
    flagged = [item for item in audit["records"] if item["flags"]]
    flagged.sort(key=lambda item: (-item["risk_score"], item["split"], item["producer_slug"], item["title"]))
    lines = [
        "# Catalog Risk Audit",
        "",
        "This audit flags catalog entries that deserve review before future expansion. A flag is not an automatic removal decision.",
        "",
        "## Summary",
        "",
        f"- Records audited: {audit['summary']['records']}",
        f"- Flagged records: {audit['summary']['flagged_records']}",
        f"- High-risk records: {audit['summary']['high_risk_records']}",
        "",
        "## Flag Counts",
        "",
        "| Flag | Count |",
        "|---|---:|",
    ]
    for code, count in audit["summary"]["flag_counts"].items():
        lines.append(f"| `{code}` | {count} |")
    lines.extend([
        "",
        "## Review List",
        "",
        "| Severity | Split | Producer | Title | VocaDB | Flags |",
        "|---|---|---|---|---:|---|",
    ])
    for item in flagged[:160]:
        highest = min(
            (flag["severity"] for flag in item["flags"]),
            key=lambda severity: severity_order.get(severity, 99),
        )
        flags = "<br>".join(
            f"`{flag['code']}`: {flag['detail']}"
            for flag in sorted(item["flags"], key=lambda flag: severity_order.get(flag["severity"], 99))
        )
        title = str(item.get("title") or item.get("vocadb_name") or "").replace("|", "\\|")
        lines.append(
            "| "
            f"{highest} | {item['split']} | {item['producer_slug']} | {title} | "
            f"{item.get('vocadb_song_id') or ''} | {flags} |"
        )
    if len(flagged) > 160:
        lines.append(f"\nOnly the first 160 flagged records are shown. See `{audit['output_json']}` for the full audit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "evaluations" / "catalog_risk_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=root / "docs" / "CATALOG_RISK_AUDIT.md",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw_jsonl" / "vocadb_song_cache",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    producers = producer_map(root)
    records = collect_records(root)
    session = requests.Session()
    session.headers["User-Agent"] = "VocaPTest/0.1 catalog risk audit"
    audited = []
    for index, record in enumerate(records, start=1):
        song = None
        if record.get("vocadb_song_id") is not None:
            try:
                song = fetch_song(
                    session,
                    int(record["vocadb_song_id"]),
                    args.cache_dir,
                    not args.no_cache,
                )
            except requests.RequestException as exc:
                audited.append({
                    **record,
                    "flags": [{
                        "code": "vocadb_fetch_failed",
                        "severity": "high",
                        "detail": str(exc),
                    }],
                    "risk_score": 100,
                })
                continue
        audited.append(analyze_record(record, song, producers))
        if index % 50 == 0:
            print(f"audited {index}/{len(records)}")

    flag_counts = Counter(
        flag["code"]
        for item in audited
        for flag in item["flags"]
    )
    high_risk = [
        item for item in audited
        if any(flag["severity"] == "high" for flag in item["flags"])
    ]
    audit = {
        "protocol": {
            "purpose": "Catalog risk audit for producer expansion",
            "vocab_source": VOCADB_API,
        },
        "summary": {
            "records": len(audited),
            "flagged_records": sum(bool(item["flags"]) for item in audited),
            "high_risk_records": len(high_risk),
            "flag_counts": dict(sorted(flag_counts.items())),
        },
        "records": audited,
        "output_json": args.output.as_posix(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report_output, audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
