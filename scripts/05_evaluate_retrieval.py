#!/usr/bin/env python
"""Evaluate retrieval: run search on a test audio file."""
import argparse
import json
from pathlib import Path

from vocaptest.retrieval.search import ProducerSearch
from vocaptest.retrieval.build_profiles import load_profiles
from vocaptest.utils.config import load_config
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a test file")
    parser.add_argument("--profile", required=True, help="Path to profiles pickle file")
    parser.add_argument("--input", required=True, help="Path to input audio WAV")
    parser.add_argument("--config", default="configs/model_mert.yaml", help="Model config")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    # Load profiles
    profiles = load_profiles(args.profile)
    logger.info("Loaded profiles: %d producers", len(profiles["producers"]))

    # Load embedder
    cfg = load_config(args.config)
    backend = profiles.get("backend", cfg.model.get("backend", "mert_95"))

    if "muq" in backend:
        from vocaptest.models.muq_embedder import MuQEmbedder
        embedder = MuQEmbedder(
            model_name=cfg.model.get("hf_name", "OpenMuQ/MuQ-large-msd-iter"),
            device=args.device,
        )
    else:
        from vocaptest.models.mert_embedder import MERTEmbedder
        embedder = MERTEmbedder(
            model_name=cfg.model.get("hf_name", "m-a-p/MERT-v1-95M"),
            device=args.device,
        )

    # Search
    engine = ProducerSearch(
        embedder=embedder,
        profiles=profiles,
        config={"top_k": args.top_k},
    )

    results = engine.search_file(args.input)

    print(f"\n{'='*50}")
    print(f"  Top-{args.top_k} Results")
    print(f"{'='*50}")
    for r in results:
        bar = "█" * int(r.score * 40)
        print(f"  {r.rank}. {r.display_name:<20s} {r.score:.4f} {bar}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
