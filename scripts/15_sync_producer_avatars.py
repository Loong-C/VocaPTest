#!/usr/bin/env python
"""Download normalized local producer avatars from configured VocaDB profiles."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import requests
import yaml
from PIL import Image, ImageOps

from vocaptest.utils.paths import project_root


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "producers.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "web" / "public" / "avatars",
    )
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Only sync the selected producer slug; may be repeated.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "VocaPTest/0.1 avatar sync"

    producers = config.get("producers", [])
    if args.slug:
        requested = set(args.slug)
        producers = [
            producer
            for producer in producers
            if producer["slug"] in requested
        ]
        missing = requested.difference(
            producer["slug"]
            for producer in producers
        )
        if missing:
            raise ValueError(f"Unknown producer slug(s): {sorted(missing)}")

    for producer in producers:
        artist_id = producer["vocadb_artist_id"]
        response = session.get(
            f"https://vocadb.net/api/artists/{artist_id}",
            params={"fields": "MainPicture"},
            timeout=30,
        )
        response.raise_for_status()
        picture = response.json().get("mainPicture")
        if not picture:
            raise ValueError(f"VocaDB artist {artist_id} has no main picture")

        image_response = session.get(picture["urlOriginal"], timeout=30)
        image_response.raise_for_status()
        with Image.open(BytesIO(image_response.content)) as source:
            avatar = ImageOps.fit(
                source.convert("RGB"),
                (args.size, args.size),
                method=Image.Resampling.LANCZOS,
            )
            output_path = args.output / f"{producer['slug']}.webp"
            avatar.save(output_path, "WEBP", quality=88, method=6)
        print(f"{producer['slug']}: {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()
