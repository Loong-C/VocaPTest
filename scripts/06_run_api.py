#!/usr/bin/env python
"""Start the FastAPI server for local use."""
import argparse

import uvicorn

from vocaptest.utils.config import load_config
from vocaptest.utils.paths import project_root


def main():
    parser = argparse.ArgumentParser(description="Start the VocaPTest API server.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable development hot reload. Do not use for production.",
    )
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / "configs" / "api.yaml")
    api_cfg = cfg.api

    uvicorn.run(
        "vocaptest.api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
