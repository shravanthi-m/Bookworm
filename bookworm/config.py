"""Load YAML config, layering a user file over config/default.yaml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    with open(DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_PATH:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
