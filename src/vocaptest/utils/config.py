"""Configuration loader with YAML support and env override."""
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Load and merge YAML configs. Supports dict-style access."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        val = self._data.get(name)
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return self._data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def merge(cls, *configs: "Config") -> "Config":
        """Deep merge later configs over earlier ones."""
        merged: dict = {}
        for cfg in configs:
            _deep_update(merged, cfg.to_dict())
        return cls(merged)


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(*paths: str | Path) -> Config:
    """Load and merge YAML configs. Later paths override earlier ones."""
    configs = [Config.from_yaml(p) for p in paths]
    return Config.merge(*configs)
