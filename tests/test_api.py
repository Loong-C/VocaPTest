"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from vocaptest.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_list_producers():
    response = client.get("/api/producers")
    assert response.status_code == 200
    data = response.json()
    assert "producers" in data
    assert "backend" in data


def test_get_producer_not_found():
    response = client.get("/api/producers/nonexistent")
    assert response.status_code == 404


def test_analyze_no_file():
    response = client.post("/api/analyze")
    assert response.status_code == 422  # validation error


def test_job_status():
    response = client.get("/api/jobs/test_123")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test_123"
