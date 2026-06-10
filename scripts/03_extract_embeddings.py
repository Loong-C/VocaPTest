#!/usr/bin/env python
"""Extract embeddings from preprocessed audio segments."""
import argparse
import json
from pathlib import Path

from vpstyle.features.extract_embeddings import extract_embeddings
from vpstyle.data.metadata_schema import Segment
from vpstyle.utils.config import load_config
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Extract audio embeddings")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--segments", required=True, help="Path to segments manifest JSONL")
    parser.add_argument("--output", required=True, help="Output directory for embeddings")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip existing embeddings")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = cfg.model.get("backend", "mert_95")

    # Load embedder
    if backend == "muq":
        from vpstyle.models.muq_embedder import MuQEmbedder
        embedder = MuQEmbedder(
            model_name=cfg.model.get("hf_name", "OpenMuQ/MuQ-large-msd-iter"),
            device=args.device,
        )
    else:
        from vpstyle.models.mert_embedder import MERTEmbedder
        embedder = MERTEmbedder(
            model_name=cfg.model.get("hf_name", "m-a-p/MERT-v1-95M"),
            device=args.device,
            layer_strategy=cfg.model.get("layer_strategy", "mean_last_hidden"),
        )

    # Load segments
    segments = []
    with open(args.segments, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(Segment(**json.loads(line)))

    logger.info("Loaded %d segments", len(segments))

    # Extract embeddings
    output_dir = Path(args.output)
    manifest_path = output_dir / "segments.jsonl"

    records = extract_embeddings(
        segments,
        embedder,
        output_dir,
        manifest_path,
        resume=not args.no_resume,
    )

    # Save config snapshot
    snapshot_path = output_dir / "config_snapshot.yaml"
    import yaml
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)

    logger.info("Done: %d embeddings saved to %s", len(records), output_dir)


if __name__ == "__main__":
    main()
