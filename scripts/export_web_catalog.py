"""Export public producer metadata so the site works without the GPU computer."""
import argparse
import asyncio
from pathlib import Path

from vocaptest.api.routes_metadata import get_producer, list_producers


async def export(destination):
    destination.mkdir(parents=True, exist_ok=True)
    listing = await list_producers()
    if listing.total_producers != 50:
        raise RuntimeError("Expected the complete production reference library")
    (destination / "producers.json").write_text(listing.model_dump_json(), encoding="utf-8")
    details = destination / "producers"
    details.mkdir(exist_ok=True)
    for producer in listing.producers:
        item = await get_producer(producer.slug)
        (details / f"{producer.slug}.json").write_text(item.model_dump_json(), encoding="utf-8")
    print(f"Exported {listing.total_producers} producers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("web/dist/catalog"))
    asyncio.run(export(parser.parse_args().output))
