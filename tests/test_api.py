"""Tests for API endpoints."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vocaptest.api.main import app
from vocaptest.api import routes_metadata, routes_upload

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
    assert data["style_tags"]
    assert data["style_tag_source"] == "VocaDB song tags"
    assert data["style_tag_source_url"] == "https://vocadb.net/Ar/53"
    assert data["song_count"] == len(data["songs"]) == 10
    assert data["song_count"] == len(data["training_songs"]) == 10
    assert data["dev_song_count"] == len(data["dev_songs"]) >= 1
    assert data["frozen_song_count"] == len(data["frozen_songs"]) >= 1
    assert data["test_song_count"] == len(data["test_songs"])
    assert data["test_song_count"] == data["frozen_song_count"]
    allowed_media_hosts = (
        "https://www.youtube.com/",
        "https://www.nicovideo.jp/",
    )
    assert all(
        song["source_url"].startswith(allowed_media_hosts)
        for song in data["training_songs"] + data["dev_songs"] + data["frozen_songs"]
    )


def test_analyze_no_file():
    response = client.post("/api/analyze")
    assert response.status_code == 422  # validation error


def test_create_analysis_job_and_poll_status(monkeypatch):
    def fake_run_analysis_job(_job_id: str, tmp_path: str) -> None:
        Path(tmp_path).unlink(missing_ok=True)

    monkeypatch.setattr(routes_upload, "_run_analysis_job", fake_run_analysis_job)

    response = client.post(
        "/api/analyze/jobs",
        files={"file": ("test.wav", b"RIFF....WAVE", "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("job_")
    assert data["status"] == "processing"
    assert data["stage"] == "received"
    assert data["progress"] > 0

    poll_response = client.get(f"/api/jobs/{data['job_id']}")
    assert poll_response.status_code == 200
    poll_data = poll_response.json()
    assert poll_data["job_id"] == data["job_id"]
    assert poll_data["stage"] == "received"


def test_analyze_failure_returns_generic_error(monkeypatch):
    def fail_analysis(_tmp_path: str, _job_id: str, _progress_callback=None):
        raise RuntimeError("internal model path /srv/vocaptest/private.pkl")

    monkeypatch.setattr(routes_upload, "_analyze_temp_file", fail_analysis)

    response = client.post(
        "/api/analyze",
        files={"file": ("test.wav", b"RIFF....WAVE", "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["error"] == routes_upload.USER_ANALYSIS_ERROR
    assert "private.pkl" not in data["error"]


def test_job_status():
    response = client.get("/api/jobs/test_123")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test_123"
    assert data["status"] == "not_found"
