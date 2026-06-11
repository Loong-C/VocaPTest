"""Auditable curation of embedding manifests at the song/work level."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.features.extract_embeddings import resolve_embedding_path
from vocaptest.utils.paths import project_root


@dataclass(frozen=True)
class CurationDecision:
    song_id: str
    producer_slug: str
    title: str
    status: str
    category: str
    reason: str
    work_id: str
    canonical_song_id: str
    segment_count: int


@dataclass
class CurationResult:
    records: list[EmbeddingRecord]
    decisions: list[CurationDecision]
    summary: dict


def load_embedding_manifest(path: str | Path) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(EmbeddingRecord(**json.loads(line)))
    return records


def load_song_titles(path: str | Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["song_id"]: row["title"]
            for row in csv.DictReader(handle)
        }


def load_curation_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def curate_embedding_records(
    records: list[EmbeddingRecord],
    titles: dict[str, str],
    config: dict,
) -> CurationResult:
    """Apply explicit exclusions and work-level canonicalization."""
    by_song: dict[str, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        by_song[record.song_id].append(record)

    exclusions = config.get("exclude_songs", {})
    canonical_only = config.get("curation", {}).get("canonical_only", True)
    member_to_group: dict[str, dict] = {}
    for group in config.get("work_groups", []):
        canonical = group["canonical_song_id"]
        members = group["members"]
        if canonical not in members:
            raise ValueError(f"Canonical song {canonical} is not in work group {group['work_id']}")
        for song_id in members:
            if song_id in member_to_group:
                raise ValueError(f"Song {song_id} appears in multiple work groups")
            member_to_group[song_id] = group

    unknown_ids = (set(exclusions) | set(member_to_group)) - set(by_song)
    if unknown_ids:
        raise ValueError(f"Curation config references unknown song IDs: {sorted(unknown_ids)}")

    root = project_root()
    curated_records: list[EmbeddingRecord] = []
    decisions: list[CurationDecision] = []

    for song_id, song_records in sorted(by_song.items()):
        producer_slug = song_records[0].producer_slug
        if any(record.producer_slug != producer_slug for record in song_records):
            raise ValueError(f"Song {song_id} has conflicting producer labels")

        title = titles.get(song_id, song_id)
        group = member_to_group.get(song_id)
        work_id = group["work_id"] if group else song_id
        canonical_song_id = group["canonical_song_id"] if group else song_id

        if song_id in exclusions:
            exclusion = exclusions[song_id]
            status = "excluded"
            category = exclusion["category"]
            reason = exclusion["reason"]
        elif group and canonical_only and song_id != canonical_song_id:
            status = "excluded"
            category = "duplicate_work"
            reason = group.get("reason", "Non-canonical recording of the same work.")
        else:
            status = "accepted"
            category = "canonical" if group else "unflagged"
            reason = group.get("reason", "") if group else ""

        decisions.append(CurationDecision(
            song_id=song_id,
            producer_slug=producer_slug,
            title=title,
            status=status,
            category=category,
            reason=reason,
            work_id=work_id,
            canonical_song_id=canonical_song_id,
            segment_count=len(song_records),
        ))

        if status != "accepted":
            continue

        for record in song_records:
            resolved_path = resolve_embedding_path(record)
            try:
                portable_path = resolved_path.relative_to(root).as_posix()
            except ValueError:
                portable_path = str(resolved_path)
            curated_records.append(replace(
                record,
                embedding_path=portable_path,
                work_id=work_id,
                recording_id=song_id,
                title=title,
            ))

    accepted_decisions = [item for item in decisions if item.status == "accepted"]
    songs_per_producer = Counter(item.producer_slug for item in accepted_decisions)
    segments_per_producer = Counter(record.producer_slug for record in curated_records)
    minimum = config.get("curation", {}).get("minimum_songs_per_producer", 1)
    too_small = {
        slug: count for slug, count in songs_per_producer.items()
        if count < minimum
    }
    if too_small:
        raise ValueError(
            f"Curated classes below minimum of {minimum} songs: {too_small}"
        )

    summary = {
        "input_songs": len(by_song),
        "accepted_songs": len(accepted_decisions),
        "excluded_songs": len(decisions) - len(accepted_decisions),
        "accepted_segments": len(curated_records),
        "producer_count": len(songs_per_producer),
        "songs_per_producer": dict(sorted(songs_per_producer.items())),
        "segments_per_producer": dict(sorted(segments_per_producer.items())),
        "exclusions_by_category": dict(sorted(Counter(
            item.category for item in decisions if item.status == "excluded"
        ).items())),
    }
    return CurationResult(curated_records, decisions, summary)


def write_curation_result(result: CurationResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "segments.jsonl", "w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with open(output_dir / "song_decisions.jsonl", "w", encoding="utf-8") as handle:
        for decision in result.decisions:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)
