"""The four code paths, one per combination of agent count and control style. Policy mode and strategy are
config values, so each class serves two setups and all their variants."""
from __future__ import annotations

from typing import Any

from SupportOpsBot.runtime.setups.base import Setup
from SupportOpsBot.runtime.setups.multi_agentic import MultiAgentic
from SupportOpsBot.runtime.setups.multi_graph import MultiGraph
from SupportOpsBot.runtime.setups.single_agentic import SingleAgentic
from SupportOpsBot.runtime.setups.single_graph import SingleGraph
from SupportOpsBot.runtime.shared.config import load_config

SETUP_CLASSES: dict[tuple[str, str], type[Setup]] = {
    (cls.agents, cls.control): cls for cls in (SingleAgentic, SingleGraph, MultiAgentic, MultiGraph)}


def build_setup(setup_id: str | None = None, cfg: dict[str, Any] | None = None) -> Setup:
    """The setup for `setup_id` (e.g. "s3-c"), or for an already resolved `cfg`."""
    if cfg is None:
        cfg = load_config(setup=setup_id)
    elif setup_id is not None and cfg["setup"]["id"] != setup_id:
        raise ValueError(f"cfg is resolved for {cfg['setup']['id']}, not {setup_id}")
    return SETUP_CLASSES[(cfg["setup"]["agents"], cfg["setup"]["control"])](cfg)
