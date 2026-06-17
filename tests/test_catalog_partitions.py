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
P2_WEAK_TARGETS = {
    "neru",
    "jin",
    "nakiso",
    "surii",
    "r_sound_design",
}


def load_yaml_songs(name: str) -> list[dict]:
    with open(ROOT / "configs" / name, "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("songs", [])


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_31_producers_and_balanced_new_training_catalog():
    with open(
        ROOT / "configs" / "producers.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        producers = (yaml.safe_load(handle) or {}).get("producers", [])
    additions = load_yaml_songs("training_catalog_additions.yaml")
    counts = Counter(item["producer_slug"] for item in additions)

    assert len(producers) == 31
    assert NEW_PRODUCERS.issubset(item["slug"] for item in producers)
    assert P2_PRODUCERS.issubset(item["slug"] for item in producers)
    assert {
        slug: counts[slug]
        for slug in NEW_PRODUCERS - P2_WEAK_TARGETS
    } == {
        slug: 10 for slug in NEW_PRODUCERS - P2_WEAK_TARGETS
    }
    assert {slug: counts[slug] for slug in P2_PRODUCERS} == {
        slug: 10 for slug in P2_PRODUCERS
    }


def test_holdout_catalogs_have_disjoint_songs_per_producer():
    training = load_yaml_songs("training_catalog_additions.yaml")
    dev = load_yaml_songs("dev_holdout_catalog.yaml")
    frozen = load_yaml_songs("frozen_test_catalog.yaml")
    dev_counts = Counter(item["producer_slug"] for item in dev)
    counts = Counter(item["producer_slug"] for item in frozen)

    assert len(dev) == 62
    assert set(dev_counts.values()) == {2}
    assert len(frozen) == 124
    assert set(counts.values()) == {4}
    assert not (
        {item["youtube_id"] for item in training}
        & {item["youtube_id"] for item in dev}
    )
    assert not (
        {item["youtube_id"] for item in training}
        & {item["youtube_id"] for item in frozen}
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
        {item["youtube_id"] for item in dev}
        & {item["youtube_id"] for item in frozen}
    )
    assert not (
        {int(item["vocadb_song_id"]) for item in dev}
        & {int(item["vocadb_song_id"]) for item in frozen}
    )


def test_materialized_training_and_frozen_manifests_do_not_overlap():
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

    assert len(training) == 376
    assert len(dev) == 62
    assert len(frozen) == 124
    materialized_counts = Counter(item["producer_slug"] for item in training)
    assert {slug: materialized_counts[slug] for slug in P2_WEAK_TARGETS} == {
        slug: 16 for slug in P2_WEAK_TARGETS
    }
    assert {slug: materialized_counts[slug] for slug in P2_PRODUCERS} == {
        slug: 10 for slug in P2_PRODUCERS
    }
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
    assert "VocaDB-listed Original YouTube PV" in original_pv
    assert "official YouTube upload" not in original_pv
    assert "non-original YouTube PV" in source_reason("vocadb_other_pv")
