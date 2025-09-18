# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for orchestrator integration tests."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from orchestrator.actions.base import Action
from orchestrator.agents import register as register_agent
from orchestrator.agents.base import Agent


class CaptureAgent(Agent):
    """Agent used in tests to capture orchestrator outputs."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, **kwargs)
        self.events = []

    async def handle(self, *, message, payload, latency_ms: float):  # type: ignore[override]
        event = {
            "message": message,
            "payload": payload,
            "latency_ms": latency_ms,
        }
        self.events.append(event)
        return []


@pytest.fixture(scope="session", autouse=True)
def register_test_agents():
    """Register the capture agent type for tests."""
    try:
        register_agent("capture", lambda name, **opts: CaptureAgent(name, **opts))
    except ValueError:
        # Already registered, e.g. when tests rerun in the same interpreter.
        pass
    yield
