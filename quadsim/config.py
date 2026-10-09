"""Config loading: YAML file + dotted-key overrides (e.g. physics.rtf=0)."""
from __future__ import annotations

import json
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "sim.yaml")


def _set(cfg: dict, dotted: str, value):
    node = cfg
    *path, last = dotted.split(".")
    for k in path:
        node = node.setdefault(k, {})
    node[last] = value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_scene_meta(scene_dir: str) -> dict:
    with open(os.path.join(scene_dir, "meta.json")) as f:
        return json.load(f)
