"""Config loading: base.yaml, then an experiment file, then dotted CLI overrides.
Also loads secrets (API keys) from .env into the environment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: str | Path = ROOT / ".env") -> bool:
    """Read KEY=value lines from .env into os.environ. Variables already set in the shell
    win over .env. Returns False if there is no .env file."""
    return load_dotenv(path, override=False)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    load_env()
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    cfg = base
    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        exp = yaml.safe_load(path.read_text()) or {}
        exp.pop("extends", None)
        cfg = deep_merge(base, exp)
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)
    return cfg
