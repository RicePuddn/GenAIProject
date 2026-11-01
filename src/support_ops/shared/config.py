"""Config loading: base.yaml, an optional extra file (e.g. a RAG profile), the setup, then dotted overrides.
Also loads secrets (API keys) from .env into the environment.

A setup ID such as `s3-c` resolves from configs/setups.yaml into `setup` (its name, agents, control style
and variant), `policy.mode` and `strategy` (each role's react or plan_execute).
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
SETUPS_FILE = ROOT / "configs" / "setups.yaml"
SETUP_ID = re.compile(r"(s[1-8])-([a-d])")


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


def _matrix() -> dict[str, Any]:
    return yaml.safe_load(SETUPS_FILE.read_text())


def setup_ids() -> list[str]:
    """Every setup ID, s1-a to s8-d, in order: 24 variants."""
    matrix = _matrix()
    return [f"{name.lower()}-{variant}" for name, setup in matrix["setups"].items()
            for variant in matrix["variants"][setup["agents"]]]


def resolve_setup(setup_id: str) -> dict[str, Any]:
    """The config a setup ID adds to base.yaml."""
    match = SETUP_ID.fullmatch(str(setup_id).strip().lower())
    if match is None:
        raise ValueError(f"unknown setup {setup_id!r}: expected an ID such as s1-a or s3-c")
    matrix = _matrix()
    name, variant = match[1].upper(), match[2]
    setup = matrix["setups"][name]
    variants = matrix["variants"][setup["agents"]]
    if variant not in variants:
        raise ValueError(f"{name} has variants {', '.join(variants)}, not {variant}")
    return {
        "setup": {"id": f"{name.lower()}-{variant}", "name": name, "variant": variant, "agents": setup["agents"],
                  "control": setup["control"], "owner": setup["owner"]},
        "policy": {"mode": setup["policy"]},
        "strategy": dict(variants[variant]),
    }


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None, *,
                setup: str | None = None) -> dict:
    """base.yaml, then `path` or SUPPORT_OPS_CONFIG, then setup resolution and dotted overrides."""
    load_env()
    if path is None:
        path = os.environ.get("SUPPORT_OPS_CONFIG")
    overrides = dict(overrides or {})
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    if path is not None:
        path = Path(path)
        extra = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text()) or {}
        cfg = deep_merge(cfg, extra)
    cfg = deep_merge(cfg, resolve_setup(setup or overrides.pop("run.setup", None) or cfg["run"]["setup"]))
    cfg["run"]["setup"] = cfg["setup"]["id"]
    for dotted, value in overrides.items():
        set_dotted(cfg, dotted, value)
    return cfg
