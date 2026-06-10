#!/usr/bin/env python
"""Start the FastAPI dev server."""
import uvicorn

from vpstyle.utils.config import load_config
from vpstyle.utils.paths import project_root


def main():
    root = project_root()
    cfg = load_config(root / "configs" / "api.yaml")
    api_cfg = cfg.api

    uvicorn.run(
        "vpstyle.api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=True,
    )


if __name__ == "__main__":
    main()
