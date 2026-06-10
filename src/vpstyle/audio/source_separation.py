"""Source separation stub for future use.

When source separation is needed (for distinguishing "composition style"
vs "vocal tuning style" vs "arrangement style"), integrate a model like
demucs or spleeter here.
"""
from __future__ import annotations

from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def separate_sources(wav_path: str, output_dir: str) -> dict[str, str]:
    """Placeholder for source separation.

    Returns a dict mapping stem name to output path.
    Not implemented in MVP.
    """
    logger.warning("Source separation is not implemented in MVP.")
    return {}
