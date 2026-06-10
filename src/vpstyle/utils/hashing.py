"""Content-based hashing for audio and manifest records."""
import hashlib
from pathlib import Path


def file_hash(path: str | Path, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def str_hash(text: str, algorithm: str = "sha256") -> str:
    """Compute hex digest of a string (useful for generating segment IDs)."""
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()
