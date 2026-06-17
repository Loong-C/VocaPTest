"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from vocaptest.api.main import app
from vocaptest.api import routes_metadata

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}


def test_list_producers():
    response = client.get("/api/producers")
    assert response.status_code == 200
    data = response.json()
    assert "producers" in data
    assert "backend" in data


def test_get_producer_not_found():
    response = client.get("/api/producers/nonexistent")
    assert response.status_code == 404


def test_get_producer_includes_metadata_and_training_songs(monkeypatch):
    monkeypatch.setattr(
        routes_metadata,
        "get_reference_library",
        lambda: {
            "backend": "test",
            "producers": {
                "wowaka": {
                    "display_name": "wowaka",
                    "song_count": 5,
                    "segment_count": 60,
                }
            },
        },
    )

    response = client.get("/api/producers/wowaka")

    assert response.status_code == 200
    data = response.json()
    assert data["avatar_url"] == "/avatars/wowaka.webp"
    assert "現実逃避P" in data["aliases"]
    assert data["song_count"] == len(data["songs"]) == 10
    assert data["song_count"] == len(data["training_songs"]) == 10
    assert data["dev_song_count"] == len(data["dev_songs"]) == 2
    assert data["frozen_song_count"] == len(data["frozen_songs"]) == 4
    assert data["test_song_count"] == len(data["test_songs"]) == 4
    assert all(
        song["source_url"].startswith("https://www.youtube.com/")
        for song in data["training_songs"] + data["dev_songs"] + data["frozen_songs"]
    )


def test_analyze_no_file():
    response = client.post("/api/analyze")
    assert response.status_code == 422  # validation error


def test_job_status():
    response = client.get("/api/jobs/test_123")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test_123"
