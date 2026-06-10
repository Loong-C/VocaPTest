"""CLAP embedder stub — placeholder for future use."""
from __future__ import annotations

import numpy as np

from vpstyle.models.base import AudioEmbedder
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


class CLAPEmbedder(AudioEmbedder):
    """CLAP-based audio embedder (stub). Not used in MVP."""

    def __init__(self, model_name: str = "", device: str = "cuda"):
        self._device = device
        self._dim = 512
        self._sr = 48000
        self._backend = "clap"
        logger.warning(
            "CLAPEmbedder is a stub — not implemented in MVP."
        )

    def embed_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError("CLAP embedder not implemented in MVP")

    def embed_file(self, wav_path: str) -> np.ndarray:
        raise NotImplementedError("CLAP embedder not implemented in MVP")

    def embed_batch(self, wavs: list[np.ndarray], sr: int) -> np.ndarray:
        raise NotImplementedError("CLAP embedder not implemented in MVP")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def backend_name(self) -> str:
        return self._backend
