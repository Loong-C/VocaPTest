"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vpstyle.api.routes_upload import router as upload_router
from vpstyle.api.routes_search import router as search_router
from vpstyle.api.routes_metadata import router as metadata_router
from vpstyle.api.schemas import HealthResponse
from vpstyle.api.dependencies import get_config, get_profiles
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Vocaloid Producer Style API...")
    cfg = get_config()
    logger.info("Config loaded: backend=%s, top_k=%d",
                 cfg.model.get("backend", "unknown"),
                 cfg.retrieval.get("top_k", 5))
    # Pre-load profiles on startup
    profiles = get_profiles()
    logger.info("API ready: %d producers loaded", len(profiles.get("producers", {})))
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Vocaloid Producer Style API",
    description="Upload music and discover which Vocaloid producer your style resembles most.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
cfg = get_config()
origins = cfg.api.get("cors_origins", ["http://localhost:3000", "http://localhost:5173"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(metadata_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    profiles = get_profiles()
    return HealthResponse(
        status="ok",
        backend=profiles.get("backend"),
        producers_loaded=len(profiles.get("producers", {})),
    )
