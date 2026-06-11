"""Abstract base for all audio embedders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioEmbedder(ABC):
    """Unified interface for audio embedding models (MERT, MuQ, CLAP, etc.)."""

    @abstractmethod
    def embed_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Extract embedding from a waveform array.

        Args:
            wav: 1-D numpy array of audio samples.
            sr: Sample rate of the waveform.

        Returns:
            1-D numpy array (embedding_dim,).
        """
        ...

    @abstractmethod
    def embed_file(self, wav_path: str) -> np.ndarray:
        """Extract embedding from an audio file.

        Args:
            wav_path: Path to an audio file.

        Returns:
            1-D numpy array (embedding_dim,).
        """
        ...

    @abstractmethod
    def embed_batch(self, wavs: list[np.ndarray], sr: int) -> np.ndarray:
        """Extract embeddings from a batch of waveforms.

        Args:
            wavs: List of 1-D numpy arrays.
            sr: Sample rate.

        Returns:
            2-D numpy array (batch_size, embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected input sample rate."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier (e.g. 'mert_95')."""
        ...
