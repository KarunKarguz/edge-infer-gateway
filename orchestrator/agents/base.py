# SPDX-License-Identifier: Apache-2.0
"""Agent base classes for decision logic around inference outputs."""
from __future__ import annotations

import abc
from typing import Iterable, List

from orchestrator.actions.base import Action


class Agent(abc.ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.options = kwargs

    async def start(self) -> None:
        """Optional async initialisation."""

    async def stop(self) -> None:
        """Optional async teardown."""

    @abc.abstractmethod
    async def handle(self, *, message, payload, latency_ms: float) -> Iterable[Action]:
        raise NotImplementedError


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, Agent] = {}

    def register(self, name: str, agent: Agent) -> None:
        if name in self._agents:
            raise ValueError(f"agent '{name}' already registered")
        self._agents[name] = agent

    def get(self, name: str) -> Agent:
        return self._agents[name]

    def all(self) -> List[Agent]:
        return list(self._agents.values())
