"""MQTT action dispatcher for actuator commands."""
from __future__ import annotations

import asyncio
import logging

from asyncio_mqtt import Client

from .base import Action, BaseDispatcher

log = logging.getLogger(__name__)


class MQTTDispatcher(BaseDispatcher):
    def __init__(self, name: str, options):
        super().__init__(name, options)
        self._lock = asyncio.Lock()
        self._client: Client | None = None

    async def _ensure(self) -> Client:
        async with self._lock:
            if self._client is not None:
                return self._client
            host = self.options.get("host", "127.0.0.1")
            port = int(self.options.get("port", 1883))
            username = self.options.get("username")
            password = self.options.get("password")
            client = Client(hostname=host, port=port, username=username, password=password)
            await client.connect()
            self._client = client
            log.info("connected MQTT dispatcher %s to %s:%s", self.name, host, port)
            return client

    async def dispatch(self, action: Action, *, agent: str, pipeline: str) -> None:
        client = await self._ensure()
        topic = action.target or self.options.get("topic")
        if not topic:
            log.warning("MQTT dispatcher %s missing topic", self.name)
            return
        payload_str = self.options.get("format", "json")
        qos = int(self.options.get("qos", 0))
        retain = bool(self.options.get("retain", False))
        payload = action.payload
        if payload_str == "json":
            import json
            data = json.dumps({"agent": agent, "pipeline": pipeline, **payload}).encode("utf-8")
        else:
            data = payload if isinstance(payload, (bytes, bytearray)) else str(payload).encode("utf-8")
        await client.publish(topic, data, qos=qos, retain=retain)

    async def close(self) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None
