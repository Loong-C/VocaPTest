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


def load_yaml_songs(name: str) -> list[dict]:
    with open(ROOT / "configs" / name, "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("songs", [])


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_27_producers_and_balanced_new_training_catalog():
    with open(
        ROOT / "configs" / "producers.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        producers = (yaml.safe_load(handle) or {}).get("producers", [])
    additions = load_yaml_songs("training_catalog_additions.yaml")
    counts = Counter(item["producer_slug"] for item in additions)

    assert len(producers) == 27
    assert NEW_PRODUCERS.issubset(item["slug"] for item in producers)
    assert {slug: counts[slug] for slug in NEW_PRODUCERS} == {
        slug: 10 for slug in NEW_PRODUCERS
    }


def test_frozen_catalog_has_two_disjoint_songs_per_producer():
    training = load_yaml_songs("training_catalog_additions.yaml")
    frozen = load_yaml_songs("frozen_test_catalog.yaml")
    counts = Counter(item["producer_slug"] for item in frozen)

    assert len(frozen) == 54
    assert set(counts.values()) == {2}
    assert not (
        {item["youtube_id"] for item in training}
        & {item["youtube_id"] for item in frozen}
    )
    assert not (
        {int(item["vocadb_song_id"]) for item in training}
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

    assert len(training) == 309
    assert len(frozen) == 54
    assert not (
        {item["song_id"] for item in training}
        & {item["song_id"] for item in frozen}
    )
    assert not (
        {item["work_id"] for item in training}
        & {item["work_id"] for item in frozen}
    )


def test_source_reasons_do_not_mislabel_original_pvs_as_official_uploads():
    assert "official YouTube upload" in source_reason("official_upload")
    original_pv = source_reason("vocadb_original_pv")
    assert "VocaDB-listed Original YouTube PV" in original_pv
    assert "official YouTube upload" not in original_pv
