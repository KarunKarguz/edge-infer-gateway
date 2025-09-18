# SPDX-License-Identifier: Apache-2.0
"""Agent factory."""
from __future__ import annotations

from typing import Callable

from orchestrator.config import PipelineConfig

from .base import Agent

AGENT_TYPES: dict[str, Callable[..., Agent]] = {}


def register(agent_name: str, factory: Callable[..., Agent]) -> None:
    AGENT_TYPES[agent_name] = factory


def build_agents(agent_defs: dict[str, dict]) -> dict[str, Agent]:
    instances: dict[str, Agent] = {}
    for name, spec in agent_defs.items():
        agent_type = spec.get("type", name)
        options = {k: v for k, v in spec.items() if k != "type"}
        if agent_type not in AGENT_TYPES:
            raise ValueError(f"unknown agent type '{agent_type}'")
        instances[name] = AGENT_TYPES[agent_type](name=name, **options)
    return instances


from .threshold import ThresholdAgent
from .vision import PersonInZoneAgent, SnapshotArchiveAgent

register("threshold", lambda name, **opts: ThresholdAgent(name, **opts))
register("person_in_zone", lambda name, **opts: PersonInZoneAgent(name, **opts))
register("snapshot_archive", lambda name, **opts: SnapshotArchiveAgent(name, **opts))
