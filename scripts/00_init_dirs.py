#!/usr/bin/env python
"""Initialize project directories."""
from pathlib import Path

from vocaptest.utils.config import load_config
from vocaptest.utils.paths import ProjectPaths, project_root


def main():
    root = project_root()
    cfg = load_config(root / "configs" / "default.yaml")
    paths = ProjectPaths(root, cfg.paths.to_dict())
    paths.ensure_all()
    print(f"Directories initialized at: {root}")


if __name__ == "__main__":
    main()
