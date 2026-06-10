"""FastAPI dependencies — config, models, search engine."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from vpstyle.models.base import AudioEmbedder
from vpstyle.retrieval.build_profiles import load_profiles
from vpstyle.retrieval.search import ProducerSearch
from vpstyle.utils.config import load_config
from vpstyle.utils.logging import setup_logging
from vpstyle.utils.paths import project_root

logger = setup_logging()

# Lazy-loaded singletons
_embedder: Optional[AudioEmbedder] = None
_profiles: Optional[dict] = None
_search_engine: Optional[ProducerSearch] = None


@lru_cache
def get_config():
    """Load merged config from configs directory."""
    root = project_root()
    base = root / "configs" / "default.yaml"
    api_cfg = root / "configs" / "api.yaml"
    model_cfg = root / "configs" / "model_mert.yaml"
    retrieval_cfg = root / "configs" / "retrieval.yaml"
    return load_config(base, api_cfg, model_cfg, retrieval_cfg)


def get_embedder() -> AudioEmbedder:
    """Get or create the audio embedder singleton."""
    global _embedder
    if _embedder is None:
        cfg = get_config()
        backend = cfg.model.backend

        if backend == "muq":
            from vpstyle.models.muq_embedder import MuQEmbedder
            _embedder = MuQEmbedder(
                model_name=cfg.model.hf_name,
                device=cfg.model.get("device", "cuda"),
            )
        else:
            from vpstyle.models.mert_embedder import MERTEmbedder
            _embedder = MERTEmbedder(
                model_name=cfg.model.get("hf_name", "m-a-p/MERT-v1-95M"),
                device=cfg.model.get("device", "cuda"),
                layer_strategy=cfg.model.get("layer_strategy", "mean_last_hidden"),
            )
    return _embedder


def get_profiles() -> dict:
    """Get or load producer profiles."""
    global _profiles
    if _profiles is None:
        cfg = get_config()
        root = project_root()
        profile_path = root / cfg.paths.profile_dir / "profiles_mert_95.pkl"
        if profile_path.exists():
            _profiles = load_profiles(profile_path)
            logger.info("Profiles loaded: %d producers", len(_profiles.get("producers", {})))
        else:
            logger.warning("No profiles found at %s — using empty profiles", profile_path)
            _profiles = {"backend": "none", "producers": {}}
    return _profiles


def get_search_engine() -> ProducerSearch:
    """Get or create the search engine."""
    global _search_engine
    if _search_engine is None:
        cfg = get_config()
        _search_engine = ProducerSearch(
            embedder=get_embedder(),
            profiles=get_profiles(),
            config={
                "sample_rate": cfg.audio.get("sample_rate", 24000),
                "segment_seconds": cfg.audio.get("segment_seconds", 20.0),
                "hop_seconds": cfg.audio.get("hop_seconds", 10.0),
                "min_rms_db": cfg.audio.get("min_rms_db", -45.0),
                "max_segments_per_song": cfg.audio.get("max_segments_per_song", 12),
                "segment_top_ratio": cfg.retrieval.get("segment_top_ratio", 0.4),
                "top_k": cfg.retrieval.get("top_k", 5),
            },
        )
    return _search_engine
