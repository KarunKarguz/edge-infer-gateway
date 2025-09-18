# SPDX-License-Identifier: Apache-2.0
"""Simple threshold agent for environmental sensing."""
from __future__ import annotations

from typing import Iterable

from orchestrator.actions.base import Action

from .base import Agent


class ThresholdAgent(Agent):
    async def handle(self, *, message, payload, latency_ms: float) -> Iterable[Action]:
        metric = self.options.get("metric", "value")
        threshold = float(self.options.get("threshold", 0.5))
        current = payload.get(metric) if isinstance(payload, dict) else None
        if current is None:
            return []
        if current >= threshold:
            return [
                Action(
                    dispatcher=self.options.get("dispatcher", "log"),
                    target=self.options.get("target"),
                    payload={"metric": metric, "value": current, "threshold": threshold, "sensor": message.sensor_id},
                )
            ]
        return []
