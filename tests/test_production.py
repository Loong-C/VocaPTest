import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vocaptest.api.production import create_app
from vocaptest.api import routes_upload

KEY = "test-key-" + "x" * 32
HEADERS = {"Authorization": f"Bearer {KEY}"}
FILE = {"file": ("test.wav", b"RIFF....WAVE", "audio/wav")}


def test_requires_readiness_and_authentication_before_upload():
    app = create_app(api_key=KEY, warmup=lambda: None)
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert client.post("/api/analyze/jobs", headers=HEADERS).status_code == 503
    with client:
        assert client.get("/health").json()["device"] == "cuda"
        assert client.post("/api/analyze/jobs", files=FILE).status_code == 401
    assert not app.state.ready


def test_failed_cuda_startup_never_reports_ready():
    def fail():
        raise RuntimeError("CUDA unavailable")
    app = create_app(api_key=KEY, warmup=fail)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        with TestClient(app):
            pass
    assert not app.state.ready


def test_busy_job_is_rejected_and_health_remains_responsive(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def run(job_id, tmp_path):
        entered.set()
        assert release.wait(10)
        Path(tmp_path).unlink(missing_ok=True)
    monkeypatch.setattr(routes_upload, "_run_analysis_job", run)
    with TestClient(create_app(api_key=KEY, warmup=lambda: None)) as client:
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(client.post, "/api/analyze/jobs", files=FILE, headers=HEADERS)
            try:
                assert entered.wait(5)
                assert client.get("/health").status_code == 200
                assert client.post("/api/analyze/jobs", files=FILE, headers=HEADERS).status_code == 429
            finally:
                release.set()
            assert pending.result().status_code == 200
        # The slot is released even if an invalid upload fails.
        assert client.post("/api/analyze/jobs", headers=HEADERS).status_code == 422
        assert client.post("/api/analyze/jobs", headers=HEADERS).status_code == 422
