"""Project-wide path management based on config."""
from pathlib import Path
from typing import Optional


class ProjectPaths:
    """Resolve all project paths from a root directory and config."""

    def __init__(self, root: Path, paths_config: dict):
        self.root = Path(root)
        self._cfg = paths_config

    @property
    def raw_audio_dir(self) -> Path:
        return self.root / self._cfg.get("raw_audio_dir", "data/raw/audio/producers")

    @property
    def upload_dir(self) -> Path:
        return self.root / self._cfg.get("upload_dir", "data/raw/audio/uploads")

    @property
    def metadata_dir(self) -> Path:
        return self.root / self._cfg.get("metadata_dir", "data/raw/metadata")

    @property
    def interim_dir(self) -> Path:
        return self.root / self._cfg.get("interim_dir", "data/interim")

    @property
    def wav_dir(self) -> Path:
        return self.interim_dir / "wav_24k"

    @property
    def segments_dir(self) -> Path:
        return self.interim_dir / "segments"

    @property
    def embedding_dir(self) -> Path:
        return self.root / self._cfg.get("embedding_dir", "data/processed/embeddings")

    @property
    def profile_dir(self) -> Path:
        return self.root / self._cfg.get("profile_dir", "data/processed/producer_profiles")

    @property
    def splits_dir(self) -> Path:
        return self.root / "data/processed/splits"

    def ensure_all(self) -> None:
        """Create all directories if they do not exist."""
        for attr in [
            "raw_audio_dir", "upload_dir", "metadata_dir",
            "wav_dir", "segments_dir", "embedding_dir",
            "profile_dir", "splits_dir",
        ]:
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


def project_root(start: Optional[Path] = None) -> Path:
    """Walk upward to find the project root (contains pyproject.toml or .git)."""
    current = Path(start) if start else Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current
