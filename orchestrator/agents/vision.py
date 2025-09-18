"""Vision agents for YOLO detections."""
from __future__ import annotations

from typing import Iterable, List

from orchestrator.actions.base import Action

from .base import Agent


class PersonInZoneAgent(Agent):
    async def handle(self, *, message, payload, latency_ms: float) -> Iterable[Action]:
        if isinstance(payload, dict):
            detections = payload.get("detections", [])
        else:
            detections = payload if isinstance(payload, list) else []
        zone = self.options.get("zone")
        target = self.options.get("target")
        dispatcher = self.options.get("dispatcher", "log")
        hits: List[dict] = []
        for det in detections:
            if det.get("label") != "person":
                continue
            if zone and det.get("zone") != zone:
                continue
            hits.append(det)
        if not hits:
            return []
        return [
            Action(
                dispatcher=dispatcher,
                target=target,
                payload={"detections": hits, "latency_ms": latency_ms},
            )
        ]


class SnapshotArchiveAgent(Agent):
    async def handle(self, *, message, payload, latency_ms: float) -> Iterable[Action]:
        if not isinstance(payload, dict) or not payload.get("image"):
            return []
        dispatcher = self.options.get("dispatcher", "log")
        target = self.options.get("target")
        return [
            Action(
                dispatcher=dispatcher,
                target=target,
                payload={
                    "sensor": message.sensor_id,
                    "latency_ms": latency_ms,
                    "image": payload["image"],
                },
            )
        ]
