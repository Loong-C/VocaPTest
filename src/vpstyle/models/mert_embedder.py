"""MERT embedder via HuggingFace transformers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from vpstyle.models.base import AudioEmbedder
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


class MERTEmbedder(AudioEmbedder):
    """MERT-v1 embedding extractor via HuggingFace.

    Supports both MERT-v1-95M and MERT-v1-330M.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        device: str = "cuda",
        layer_strategy: str = "mean_last_hidden",
        trust_remote_code: bool = True,
    ):
        from transformers import AutoModel

        self._model_name = model_name
        self._device = device
        self._layer_strategy = layer_strategy

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(device)
        self.model.eval()

        # Determine embedding dimension from model config
        self._dim = self.model.config.hidden_size
        self._sr = 24000

        if "330" in model_name:
            self._backend = "mert_330"
        else:
            self._backend = "mert_95"

        logger.info(
            "Loaded %s on %s (dim=%d)",
            model_name, device, self._dim,
        )

    # ---- AudioEmbedder interface --------------------------------------------

    @torch.no_grad()
    def embed_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Extract embedding from waveform array."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )

        # Ensure correct sample rate
        if sr != self._sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self._sr)
            sr = self._sr

        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = self._select_hidden(outputs.hidden_states)
        emb = hidden.mean(dim=1)  # mean over time
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
        """Extract embeddings from a batch."""
        import librosa
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )

        if sr != self._sr:
            wavs = [librosa.resample(w, orig_sr=sr, target_sr=self._sr) for w in wavs]
            sr = self._sr

        inputs = processor(wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = self._select_hidden(outputs.hidden_states)

        # Mean pool with attention mask
        mask = inputs.get("attention_mask")
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1)
            emb = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            emb = hidden.mean(dim=1)

        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    # ---- Internal -----------------------------------------------------------

    def _select_hidden(self, hidden_states: tuple) -> torch.Tensor:
        """Select hidden states based on layer strategy."""
        if self._layer_strategy == "mean_last_hidden":
            return hidden_states[-1]
        elif self._layer_strategy == "mean_all_layers":
            stacked = torch.stack(hidden_states, dim=0)
            return stacked.mean(dim=0)
        else:
            return hidden_states[-1]

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
