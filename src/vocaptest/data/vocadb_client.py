"""VocaDB API client for fetching artist and song metadata."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

from vocaptest.utils.logging import setup_logging

logger = setup_logging()


class VocaDBClient:
    """Minimal VocaDB API wrapper."""

    BASE_URL = "https://vocadb.net/api"

    def __init__(self, user_agent: str = "vocaptest/0.1", base_url: Optional[str] = None):
        self.base_url = base_url or self.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    # ---- Artists ------------------------------------------------------------

    def search_artists(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Search for artists by name. Returns raw API items."""
        params = {
            "query": query,
            "maxResults": max_results,
            "fields": "AdditionalNames",
        }
        resp = self._get("/artists", params=params)
        return resp.get("items", [])

    def get_artist(self, artist_id: int) -> dict:
        """Get full artist details."""
        return self._get(f"/artists/{artist_id}")

    # ---- Songs --------------------------------------------------------------

    def get_songs_by_artist(
        self,
        artist_id: int,
        fields: str = "PVs,Artists,Tags",
        max_results: int = 200,
        start: int = 0,
        sort: str = "PublishDate",
        song_types: Optional[str] = None,
    ) -> list[dict]:
        """Fetch songs for a given artist. Paginated."""
        all_items: list[dict] = []
        while True:
            params: dict = {
                "artistId": artist_id,
                "fields": fields,
                "maxResults": min(max_results, 50),
                "start": start,
                "sort": sort,
            }
            if song_types:
                params["songTypes"] = song_types
            resp = self._get("/songs", params=params)
            items = resp.get("items", [])
            all_items.extend(items)
            if len(items) < 50 or len(all_items) >= max_results:
                break
            start += len(items)
            time.sleep(0.5)  # rate limit
        return all_items

    # ---- Raw cache helpers --------------------------------------------------

    def save_raw_json(self, data: list[dict], path: str | Path) -> None:
        """Write raw API results as JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved %d records to %s", len(data), path)

    # ---- Internal -----------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
