"""Main search function: input audio -> Top-K producers."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample

from vocaptest.audio.preprocess import preprocess_file
from vocaptest.audio.segment import segment_file, split_segments
from vocaptest.data.metadata_schema import SearchResult
from vocaptest.models.base import AudioEmbedder
from vocaptest.models.song_lda import SongMeanShrinkageLDA
from vocaptest.retrieval.similarity import score_song_against_all
from vocaptest.utils.logging import setup_logging

logger = setup_logging()

ProgressCallback = Callable[[str, float], None]


class ProducerSearch:
    """High-level search: embed an uploaded audio file and find similar producers."""

    def __init__(
        self,
        embedder: AudioEmbedder,
        profiles: dict | None = None,
        classifier: object | None = None,
        config: dict | None = None,
    ):
        self.embedder = embedder
        self.profiles = profiles or {}
        self.classifier = classifier
        if classifier is None and not self.profiles.get("producers"):
            raise ValueError("ProducerSearch requires a classifier or non-empty profiles")
        self._cfg = config or {}
        self._sr = self._cfg.get("sample_rate", embedder.sample_rate)
        self._segment_sec = self._cfg.get("segment_seconds", 20.0)
        self._hop_sec = self._cfg.get("hop_seconds", 10.0)
        self._min_rms_db = self._cfg.get("min_rms_db", -45.0)
        self._max_segments = self._cfg.get("max_segments_per_song", 12)
        self._segment_selection = self._cfg.get("segment_selection", "uniform")
        self._top_ratio = self._cfg.get("segment_top_ratio", 0.4)
        self._top_k = self._cfg.get("top_k", 5)
        self._inference_batch_size = self._cfg.get("inference_batch_size", 4)

    def search_file(self, audio_path: str | Path) -> list[SearchResult]:
        """Search for similar producers for a preprocessed audio file.

        The file is loaded, segmented, embedded, and scored against all profiles.
        """
        audio_path = Path(audio_path)
        wav, in_sr = sf.read(str(audio_path), dtype='float32')
        # Convert to mono if needed
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # Resample if sample rate differs
        if in_sr != self._sr:
            num_samples = int(len(wav) * self._sr / in_sr)
            wav = scipy_resample(wav, num_samples)

        return self.search_wav(wav)

    def search_file_detailed(
        self,
        audio_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[SearchResult], dict | None]:
        """Search a file and return optional calibration/rejection diagnostics."""
        audio_path = Path(audio_path)
        wav, in_sr = sf.read(str(audio_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if in_sr != self._sr:
            num_samples = int(len(wav) * self._sr / in_sr)
            wav = scipy_resample(wav, num_samples)
        return self.search_wav_detailed(wav, progress_callback=progress_callback)

    def search_wav(self, wav: np.ndarray) -> list[SearchResult]:
        """Search from an already-loaded waveform."""
        return self.search_wav_detailed(wav)[0]

    def search_wav_detailed(
        self,
        wav: np.ndarray,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[SearchResult], dict | None]:
        """Search waveform and return results plus calibrated confidence signals."""
        # Segment
        if progress_callback:
            progress_callback("segmenting", 0.25)
        segments_info = split_segments(
            wav, self._sr,
            segment_seconds=self._segment_sec,
            hop_seconds=self._hop_sec,
            min_rms_db=self._min_rms_db,
            max_segments=self._max_segments,
            selection_strategy=self._segment_selection,
        )

        if not segments_info:
            logger.warning("No valid segments found in audio")
            return [], None

        # Embed each segment
        if progress_callback:
            progress_callback("embedding", 0.45)
        chunks = [
            wav[segment["start_sample"]:segment["end_sample"]]
            for segment in segments_info
        ]
        if hasattr(self.classifier, "rank_segment_layers"):
            segment_layers = []
            total_batches = max(
                1,
                (len(chunks) + self._inference_batch_size - 1) // self._inference_batch_size,
            )
            for start in range(0, len(chunks), self._inference_batch_size):
                segment_layers.extend(self.embedder.embed_batch_layers(
                    chunks[start:start + self._inference_batch_size],
                    self._sr,
                ))
                if progress_callback:
                    batch_index = start // self._inference_batch_size + 1
                    progress_callback(
                        "embedding",
                        0.45 + 0.35 * batch_index / total_batches,
                    )
            if progress_callback:
                progress_callback("classifying", 0.85)
            prediction = self.classifier.rank_segment_layers(
                np.stack(segment_layers),
                top_k=self._top_k,
            )
            return prediction.results, {
                "accepted": prediction.accepted,
                "confidence": prediction.confidence,
                "margin": prediction.margin,
                "entropy": prediction.entropy,
            }

        segment_embs = []
        for index, chunk in enumerate(chunks, start=1):
            segment_embs.append(self.embedder.embed_wav(chunk, self._sr))
            if progress_callback:
                progress_callback("embedding", 0.45 + 0.35 * index / len(chunks))
        if not segment_embs:
            return [], None

        segment_embs = np.stack(segment_embs, axis=0)

        if progress_callback:
            progress_callback("classifying", 0.85)
        if self.classifier is not None:
            return (
                self.classifier.rank_segments(segment_embs, top_k=self._top_k),
                None,
            )

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

        return results, None
