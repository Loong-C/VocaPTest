"""MuQ embedder — requires `pip install muq`."""
from __future__ import annotations

import numpy as np
import torch

from vpstyle.models.base import AudioEmbedder
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


class MuQEmbedder(AudioEmbedder):
    """MuQ embedding extractor.

    Input: 24kHz mono waveform.
    Uses fp32 to avoid NaN issues as recommended by MuQ authors.
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        device: str = "cuda",
    ):
        try:
            from muq import MuQ
        except ImportError:
            raise ImportError(
                "MuQ is not installed. Install with: pip install muq"
            )

        self._model_name = model_name
        self._device = device
        self._sr = 24000

        self.model = MuQ.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Determine embedding dim from a test forward pass
        self._dim = self._infer_dim()

        self._backend = "muq"
        logger.info(
            "Loaded %s on %s (dim=%d)",
            model_name, device, self._dim,
        )

    def _infer_dim(self) -> int:
        """Run a dummy forward pass to determine embedding dimension."""
        dummy = torch.randn(1, self._sr * 2).float().to(self._device)
        with torch.no_grad():
            output = self.model(dummy)
        features = self._extract_features(output)
        return features.shape[-1]

    @staticmethod
    def _extract_features(output) -> torch.Tensor:
        """Extract feature tensor from MuQ output (which may be dict or tensor)."""
        if isinstance(output, dict):
            features = output.get("last_hidden_state", None)
            if features is None:
                features = output.get("x", None)
            if features is None:
                # Try first tensor value
                for v in output.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        features = v
                        break
        else:
            features = output

        if features is None:
            raise ValueError(f"Cannot extract features from MuQ output type: {type(output)}")

        return features

    # ---- AudioEmbedder interface --------------------------------------------

    @torch.no_grad()
    def embed_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Extract embedding from waveform array."""
        if sr != self._sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self._sr)

        wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(self._device)
        output = self.model(wav_tensor)
        features = self._extract_features(output)

        if features.dim() == 3:
            emb = features.mean(dim=1)
        else:
            emb = features

        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_file(self, wav_path: str) -> np.ndarray:
        """Extract embedding from an audio file."""
        import librosa
        wav, sr = librosa.load(wav_path, sr=self._sr, mono=True)
        return self.embed_wav(wav, sr)

    @torch.no_grad()
    def embed_batch(self, wavs: list[np.ndarray], sr: int) -> np.ndarray:
        """Extract embeddings from a batch (processed one-by-one for MuQ)."""
        import librosa
        results = []
        for wav in wavs:
            if sr != self._sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self._sr)
            emb = self.embed_wav(wav, self._sr)
            results.append(emb)
        return np.stack(results, axis=0)

    # ---- Properties ---------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def backend_name(self) -> str:
        return self._backend
