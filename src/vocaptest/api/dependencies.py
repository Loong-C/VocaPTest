"""FastAPI dependencies — config, models, search engine."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from vocaptest.models.base import AudioEmbedder
from vocaptest.models.layer_fusion import LayerFusionLDA
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.retrieval.build_profiles import load_profiles
from vocaptest.retrieval.search import ProducerSearch
from vocaptest.utils.config import load_config
from vocaptest.utils.logging import setup_logging
from vocaptest.utils.paths import project_root

logger = setup_logging()

# Lazy-loaded singletons
_embedder: Optional[AudioEmbedder] = None
_profiles: Optional[dict] = None
_lda_model: Optional[SongMeanShrinkageLDA] = None
_p1_model: Optional[LayerFusionLDA] = None
_search_engine: Optional[ProducerSearch] = None


def _resolve_model_device(configured_device: str | None) -> str:
    """Resolve a configured torch device, falling back safely on CPU-only hosts."""
    requested = (configured_device or "auto").strip().lower()
    if requested == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "Configured model.device=%s but CUDA is unavailable; falling back to CPU",
                configured_device,
            )
            return "cpu"

    if requested == "mps":
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            logger.warning(
                "Configured model.device=mps but MPS is unavailable; falling back to CPU",
            )
            return "cpu"

    return configured_device or "cpu"


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
        device = _resolve_model_device(cfg.model.get("device", "auto"))

        if backend == "muq":
            from vocaptest.models.muq_embedder import MuQEmbedder
            _embedder = MuQEmbedder(
                model_name=cfg.model.hf_name,
                device=device,
            )
        else:
            from vocaptest.models.mert_embedder import MERTEmbedder
            _embedder = MERTEmbedder(
                model_name=cfg.model.get("hf_name", "m-a-p/MERT-v1-95M"),
                device=device,
                layer_strategy=cfg.model.get("layer_strategy", "mean_last_hidden"),
            )
    return _embedder


def get_profiles() -> dict:
    """Get or load producer profiles."""
    global _profiles
    if _profiles is None:
        cfg = get_config()
        root = project_root()
        profile_path = root / "data" / "processed" / "profiles.pkl"
        if profile_path.exists():
            _profiles = load_profiles(profile_path)
            logger.info("Profiles loaded: %d producers", len(_profiles.get("producers", {})))
        else:
            logger.warning("No profiles found at %s — using empty profiles", profile_path)
            _profiles = {"backend": "none", "producers": {}}
    return _profiles


def get_lda_model() -> SongMeanShrinkageLDA:
    """Load the configured song-mean Shrinkage LDA artifact."""
    global _lda_model
    if _lda_model is None:
        cfg = get_config()
        root = project_root()
        configured_path = Path(cfg.retrieval.get(
            "lda_model_path",
            "data/processed/models/song_mean_shrinkage_lda.pkl",
        ))
        model_path = configured_path if configured_path.is_absolute() else root / configured_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Configured Shrinkage LDA model does not exist: {model_path}. "
                "Run scripts/07_curate_dataset.py and scripts/08_train_song_lda.py."
            )
        _lda_model = SongMeanShrinkageLDA.load(model_path)
        logger.info("Song-mean Shrinkage LDA loaded: %d producers", len(_lda_model.classes_))
    return _lda_model


def get_p1_model() -> LayerFusionLDA:
    """Load the calibrated selected-layer P1 model."""
    global _p1_model
    if _p1_model is None:
        cfg = get_config()
        root = project_root()
        configured_path = Path(cfg.retrieval.get(
            "p1_model_path",
            "data/processed/models/p1_selected_layer_lda.pkl",
        ))
        model_path = configured_path if configured_path.is_absolute() else root / configured_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Configured P1 model does not exist: {model_path}. "
                "Run scripts/09_rebuild_p1_layer_embeddings.py and "
                "scripts/13_train_p1_selected_layer.py."
            )
        _p1_model = LayerFusionLDA.load(model_path)
        logger.info(
            "P1 selected-layer LDA loaded: %d producers, layers=%s",
            len(_p1_model.classes_),
            _p1_model.layer_indices,
        )
    return _p1_model


def get_reference_library() -> dict:
    """Return metadata for the retrieval backend currently serving requests."""
    cfg = get_config()
    backend = cfg.retrieval.get("backend", "kmeans_profiles")
    if backend == "p1_selected_layer_lda":
        try:
            return get_p1_model().to_reference_library()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("%s", exc)
            return {
                "backend": "p1_selected_layer_lda_unavailable",
                "producers": {},
            }
    if backend == "song_mean_shrinkage_lda":
        try:
            return get_lda_model().to_reference_library()
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            return {
                "backend": "song_mean_shrinkage_lda_unavailable",
                "producers": {},
            }
    return get_profiles()


def get_search_engine() -> ProducerSearch:
    """Get or create the search engine."""
    global _search_engine
    if _search_engine is None:
        cfg = get_config()
        retrieval_backend = cfg.retrieval.get("backend", "kmeans_profiles")
        if retrieval_backend == "p1_selected_layer_lda":
            classifier = get_p1_model()
        elif retrieval_backend == "song_mean_shrinkage_lda":
            classifier = get_lda_model()
        else:
            classifier = None
        _search_engine = ProducerSearch(
            embedder=get_embedder(),
            profiles=None if classifier else get_profiles(),
            classifier=classifier,
            config={
                "sample_rate": cfg.audio.get("sample_rate", 24000),
                "segment_seconds": cfg.audio.get("segment_seconds", 20.0),
                "hop_seconds": cfg.audio.get("hop_seconds", 10.0),
                "min_rms_db": cfg.audio.get("min_rms_db", -45.0),
                "max_segments_per_song": cfg.audio.get("max_segments_per_song", 12),
                "segment_selection": cfg.audio.get("segment_selection", "uniform"),
                "segment_top_ratio": cfg.retrieval.get("segment_top_ratio", 0.4),
                "top_k": cfg.retrieval.get("top_k", 5),
                "inference_batch_size": cfg.retrieval.get("inference_batch_size", 4),
            },
        )
    return _search_engine
