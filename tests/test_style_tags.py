"""Tests for cached VocaDB style tag metadata."""
from __future__ import annotations

import yaml

from vocaptest.data.producer_catalog import load_producer_style_tags
from vocaptest.utils.paths import project_root


def test_vocadb_style_tags_cover_every_configured_producer():
    root = project_root()
    with open(root / "configs" / "producers.yaml", "r", encoding="utf-8") as handle:
        producers = yaml.safe_load(handle)["producers"]
    with open(root / "configs" / "producer_style_tags.yaml", "r", encoding="utf-8") as handle:
        style_config = yaml.safe_load(handle)

    configured_slugs = {producer["slug"] for producer in producers}
    tagged_slugs = set(style_config["producers"])

    assert configured_slugs == tagged_slugs
    for slug, entry in style_config["producers"].items():
        assert entry["source_url"].startswith("https://vocadb.net/Ar/")
        assert len(entry["style_tags"]) >= 3, slug
        assert all(tag.get("label") and tag.get("display_zh") for tag in entry["style_tags"])


def test_style_tag_loader_returns_display_labels_and_source():
    style_tags = load_producer_style_tags()

    assert style_tags["wowaka"]["style_tags"] == ["VOCAROCK", "摇滚", "高速感"]
    assert style_tags["wowaka"]["style_tag_source"] == "VocaDB song tags"
    assert style_tags["wowaka"]["style_tag_source_url"] == "https://vocadb.net/Ar/53"
