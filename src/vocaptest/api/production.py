"""CUDA-only, authenticated, single-job service behind the shared SSH tunnel."""
from contextlib import asynccontextmanager
import hmac
import os
import threading

from fastapi import FastAPI
from starlette.responses import JSONResponse

from vocaptest.api.dependencies import get_config, get_embedder, get_search_engine
from vocaptest.api.routes_search import router as jobs_router
from vocaptest.api.routes_upload import router as upload_router


class AdmissionMiddleware:
    """Reject before consuming the upload; hold the slot through background work."""

    def __init__(self, app, *, key: str, state):
        self.app = app
        self.key = key
        self.state = state
        self.gate = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health":
            return await self.app(scope, receive, send)
        authorization = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(authorization, f"Bearer {self.key}".encode()):
            return await JSONResponse({"detail": "unauthorized"}, 401)(scope, receive, send)
        if not self.state.ready:
            return await JSONResponse({"detail": "服务器不可用"}, 503)(scope, receive, send)
        analysis = scope["path"] in {"/api/analyze", "/api/analyze/jobs"}
        if analysis and not self.gate.acquire(blocking=False):
            return await JSONResponse(
                {"detail": "服务器正忙，请稍后重试。"}, 429, headers={"Retry-After": "3"},
            )(scope, receive, send)
        try:
            await self.app(scope, receive, send)
        finally:
            if analysis:
                self.gate.release()


def warm_cuda():
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production analysis")
    torch.set_num_threads(2)
    cfg = get_config()
    cfg.model.to_dict()["device"] = "cuda"
    # Leave headroom for Escape on the same 8 GiB GPU.
    cfg.retrieval.to_dict()["inference_batch_size"] = 1
    get_search_engine()
    embedder = get_embedder()
    if not str(embedder._device).startswith("cuda"):
        raise RuntimeError("Refusing CPU fallback")
    embedder.embed_wav(np.zeros(24000, dtype=np.float32), 24000)
    torch.cuda.synchronize()


def create_app(*, api_key: str, warmup=warm_cuda):
    if len(api_key) < 32:
        raise ValueError("A proxy credential of at least 32 characters is required")

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        warmup()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ready = False
    app.add_middleware(AdmissionMiddleware, key=api_key, state=app.state)
    app.include_router(upload_router)
    app.include_router(jobs_router)

    @app.get("/health")
    async def health():
        if not app.state.ready:
            return JSONResponse({"detail": "服务器不可用"}, 503)
        return {"status": "ok", "device": "cuda", "backend": "mert_95_p4_calibrated_stacking"}

    return app


def app_factory():
    return create_app(api_key=os.environ["VOCAP_INFERENCE_KEY"])
