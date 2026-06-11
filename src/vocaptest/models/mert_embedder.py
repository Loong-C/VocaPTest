"""MERT embedder via HuggingFace transformers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from vocaptest.models.base import AudioEmbedder
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def mean_pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    model,
) -> torch.Tensor:
    """Mean-pool frame-level hidden states with a feature-length mask."""
    if attention_mask is None:
        return hidden.mean(dim=1)

    if hasattr(model, "_get_feature_vector_attention_mask"):
        feature_mask = model._get_feature_vector_attention_mask(
            hidden.shape[1],
            attention_mask,
        )
    else:
        feature_mask = torch.nn.functional.interpolate(
            attention_mask[:, None, :].float(),
            size=hidden.shape[1],
            mode="nearest",
        ).squeeze(1)
    feature_mask = feature_mask.to(hidden.dtype)
    masked_hidden = hidden * feature_mask.unsqueeze(-1)
    denominator = feature_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked_hidden.sum(dim=1) / denominator


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
        from transformers import AutoModel, AutoProcessor

        self._model_name = model_name
        self._device = device
        self._layer_strategy = layer_strategy

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(device)
        self.model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

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
        # Ensure correct sample rate
        if sr != self._sr:
            from scipy.signal import resample
            num_samples = int(len(wav) * self._sr / sr)
            wav = resample(wav, num_samples)
            sr = self._sr

        inputs = self._processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = self._select_hidden(outputs.hidden_states)
        emb = hidden.mean(dim=1)  # mean over time
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_file(self, wav_path: str) -> np.ndarray:
        """Extract embedding from an audio file."""
        import soundfile as sf
        wav, sr = sf.read(wav_path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # convert to mono
        if sr != self._sr:
            from scipy.signal import resample
            num_samples = int(len(wav) * self._sr / sr)
            wav = resample(wav, num_samples)
        return self.embed_wav(wav, self._sr)

    @torch.no_grad()
    def embed_batch(self, wavs: list[np.ndarray], sr: int) -> np.ndarray:
        """Extract embeddings from a batch."""
        if sr != self._sr:
            from scipy.signal import resample
            wavs = [resample(w, int(len(w) * self._sr / sr)) for w in wavs]
            sr = self._sr

        inputs = self._processor(wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = self._select_hidden(outputs.hidden_states)

        # Mean pool with attention mask
        emb = mean_pool_hidden(hidden, inputs.get("attention_mask"), self.model)

        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    @torch.no_grad()
    def embed_batch_layers(
        self,
        wavs: list[np.ndarray],
        sr: int,
    ) -> np.ndarray:
        """Return time-pooled embeddings for every transformer hidden state."""
        if sr != self._sr:
            from scipy.signal import resample
            wavs = [resample(w, int(len(w) * self._sr / sr)) for w in wavs]
            sr = self._sr

        inputs = self._processor(
            wavs,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        attention_mask = inputs.get("attention_mask")
        pooled_layers = [
            mean_pool_hidden(hidden, attention_mask, self.model)
            for hidden in outputs.hidden_states
        ]
        stacked = torch.stack(pooled_layers, dim=1)
        stacked = torch.nn.functional.normalize(stacked, dim=-1)
        return stacked.cpu().numpy()

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
