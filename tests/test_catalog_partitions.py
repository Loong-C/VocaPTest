"""Regression tests for the training and frozen catalog boundary."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from vocaptest.data.catalog_sources import source_reason


ROOT = Path(__file__).resolve().parents[1]
NEW_PRODUCERS = {
    "iyowa",
    "syudou",
    "nakiso",
    "surii",
    "r_sound_design",
    "toa",
    "teniwoha",
}
P2_PRODUCERS = {
    "niru_kajitsu",
    "harumaki_gohan",
    "r_906",
    "sasakure_uk",
}
TOP100_EXPANSION_PRODUCERS = {
    "giga",
    "rerulili",
    "mikito_p",
    "hitoshizuku_p",
    "balloon",
    "kuro_usa_p",
    "mothy",
    "hiiragi_magnetite",
    "owata_p",
    "nuyuri",
}
SPARSE_DEV_PRODUCERS: set[str] = {"ryo", "eve"}
SPARSE_TRAINING_PRODUCERS: set[str] = set()


def source_identity(item: dict) -> tuple[str, str]:
    if item.get("source_service") and item.get("source_id"):
        return str(item["source_service"]), str(item["source_id"])
    if item.get("youtube_id"):
        return "Youtube", str(item["youtube_id"])
    song_id = str(item.get("song_id", ""))
    if song_id.startswith("youtube_"):
        return "Youtube", song_id.removeprefix("youtube_")
    if song_id.startswith("niconico_"):
        return "NicoNicoDouga", song_id.removeprefix("niconico_")
    raise AssertionError(f"Catalog item has no source identity: {item}")


def load_yaml_songs(name: str) -> list[dict]:
    with open(ROOT / "configs" / name, "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("songs", [])


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_producers_and_training_catalog_have_expected_coverage():
    with open(
        ROOT / "configs" / "producers.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        producers = (yaml.safe_load(handle) or {}).get("producers", [])
    additions = load_yaml_songs("training_catalog_additions.yaml")
    counts = Counter(item["producer_slug"] for item in additions)

    producer_slugs = {item["slug"] for item in producers}

    assert len(producers) >= 41
    assert NEW_PRODUCERS.issubset(producer_slugs)
    assert P2_PRODUCERS.issubset(producer_slugs)
    assert TOP100_EXPANSION_PRODUCERS.issubset(producer_slugs)
    assert not producer_slugs - set(counts)
    assert min(
        counts[slug]
        for slug in TOP100_EXPANSION_PRODUCERS - SPARSE_TRAINING_PRODUCERS
    ) >= 8
    if SPARSE_TRAINING_PRODUCERS:
        assert min(counts[slug] for slug in SPARSE_TRAINING_PRODUCERS) >= 2


def test_holdout_catalogs_have_disjoint_songs_per_producer():
    with open(
        ROOT / "configs" / "producers.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        producer_slugs = {
            item["slug"]
            for item in (yaml.safe_load(handle) or {}).get("producers", [])
        }
    training = load_yaml_songs("training_catalog_additions.yaml")
    dev = load_yaml_songs("dev_holdout_catalog.yaml")
    frozen = load_yaml_songs("frozen_test_catalog.yaml")
    dev_counts = Counter(item["producer_slug"] for item in dev)
    counts = Counter(item["producer_slug"] for item in frozen)

    assert set(dev_counts) == producer_slugs - SPARSE_DEV_PRODUCERS
    assert set(counts) == producer_slugs
    assert min(dev_counts.values()) >= 1
    assert min(counts.values()) >= 1
    assert not (
        {source_identity(item) for item in training}
        & {source_identity(item) for item in dev}
    )
    assert not (
        {source_identity(item) for item in training}
        & {source_identity(item) for item in frozen}
    )
    assert not (
        {int(item["vocadb_song_id"]) for item in training}
        & {int(item["vocadb_song_id"]) for item in dev}
    )
    assert not (
        {int(item["vocadb_song_id"]) for item in training}
        & {int(item["vocadb_song_id"]) for item in frozen}
    )
    assert not (
        {source_identity(item) for item in dev}
        & {source_identity(item) for item in frozen}
    )
    assert not (
        {int(item["vocadb_song_id"]) for item in dev}
        & {int(item["vocadb_song_id"]) for item in frozen}
    )


def test_materialized_training_and_frozen_manifests_do_not_overlap():
    with open(
        ROOT / "configs" / "producers.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        producer_slugs = {
            item["slug"]
            for item in (yaml.safe_load(handle) or {}).get("producers", [])
        }
    training = [
        item
        for item in load_jsonl(
            ROOT
            / "data"
            / "processed"
            / "curated"
            / "mert_95"
            / "song_decisions.jsonl"
        )
        if item.get("status") == "accepted"
    ]
    frozen = load_jsonl(
        ROOT / "data" / "processed" / "frozen_test" / "catalog.jsonl"
    )
    dev = load_jsonl(
        ROOT / "data" / "processed" / "dev_holdout" / "catalog.jsonl"
    )

    materialized_counts = Counter(item["producer_slug"] for item in training)
    dev_counts = Counter(item["producer_slug"] for item in dev)
    frozen_counts = Counter(item["producer_slug"] for item in frozen)
    assert set(materialized_counts) == producer_slugs
    assert set(dev_counts) == producer_slugs - SPARSE_DEV_PRODUCERS
    assert set(frozen_counts) == producer_slugs
    assert min(materialized_counts.values()) >= 2
    assert min(dev_counts.values()) >= 1
    assert min(frozen_counts.values()) >= 1
    assert not (
        {item["song_id"] for item in training}
        & {item["song_id"] for item in dev}
    )
    assert not (
        {item["work_id"] for item in training}
        & {item["work_id"] for item in dev}
    )
    assert not (
        {item["song_id"] for item in training}
        & {item["song_id"] for item in frozen}
    )
    assert not (
        {item["work_id"] for item in training}
        & {item["work_id"] for item in frozen}
    )
    assert not (
        {item["song_id"] for item in dev}
        & {item["song_id"] for item in frozen}
    )
    assert not (
        {item["work_id"] for item in dev}
        & {item["work_id"] for item in frozen}
    )


def test_source_reasons_do_not_mislabel_original_pvs_as_official_uploads():
    assert "official YouTube upload" in source_reason("official_upload")
    original_pv = source_reason("vocadb_original_pv")
    assert "VocaDB-listed Original PV" in original_pv
    assert "official YouTube upload" not in original_pv
    assert "non-original PV" in source_reason("vocadb_other_pv")
