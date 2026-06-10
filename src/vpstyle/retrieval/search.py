"""Main search function: input audio -> Top-K producers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vpstyle.audio.preprocess import preprocess_file
from vpstyle.audio.segment import segment_file, split_segments
from vpstyle.data.metadata_schema import SearchResult
from vpstyle.models.base import AudioEmbedder
from vpstyle.retrieval.similarity import score_song_against_all
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


class ProducerSearch:
    """High-level search: embed an uploaded audio file and find similar producers."""

    def __init__(
        self,
        embedder: AudioEmbedder,
        profiles: dict,
        config: dict | None = None,
    ):
        self.embedder = embedder
        self.profiles = profiles
        self._cfg = config or {}
        self._sr = self._cfg.get("sample_rate", embedder.sample_rate)
        self._segment_sec = self._cfg.get("segment_seconds", 20.0)
        self._hop_sec = self._cfg.get("hop_seconds", 10.0)
        self._min_rms_db = self._cfg.get("min_rms_db", -45.0)
        self._max_segments = self._cfg.get("max_segments_per_song", 12)
        self._top_ratio = self._cfg.get("segment_top_ratio", 0.4)
        self._top_k = self._cfg.get("top_k", 5)

    def search_file(self, audio_path: str | Path) -> list[SearchResult]:
        """Search for similar producers for a preprocessed audio file.

        The file is loaded, segmented, embedded, and scored against all profiles.
        """
        audio_path = Path(audio_path)
        import librosa
        wav, _ = librosa.load(str(audio_path), sr=self._sr, mono=True)

        return self.search_wav(wav)

    def search_wav(self, wav: np.ndarray) -> list[SearchResult]:
        """Search from an already-loaded waveform."""
        # Segment
        segments_info = split_segments(
            wav, self._sr,
            segment_seconds=self._segment_sec,
            hop_seconds=self._hop_sec,
            min_rms_db=self._min_rms_db,
            max_segments=self._max_segments,
        )

        if not segments_info:
            logger.warning("No valid segments found in audio")
            return []

        # Embed each segment
        segment_embs = []
        for seg in segments_info:
            chunk = wav[seg["start_sample"]:seg["end_sample"]]
            emb = self.embedder.embed_wav(chunk, self._sr)
            segment_embs.append(emb)

        if not segment_embs:
            return []

        segment_embs = np.stack(segment_embs, axis=0)

        # Score against profiles
        scores = score_song_against_all(
            segment_embs,
            self.profiles,
            top_ratio=self._top_ratio,
        )

        # Format results
        results = []
        for i, s in enumerate(scores[:self._top_k]):
            results.append(SearchResult(
                producer_slug=s["producer_slug"],
                display_name=s["display_name"],
                score=s["score"],
                rank=i + 1,
            ))

        return results
