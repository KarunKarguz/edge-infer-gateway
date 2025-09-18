"""MQTT connector for Wi-Fi/ESP edge nodes."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from asyncio_mqtt import Client, MqttError

from orchestrator.messages import EdgeMessage

from .base import BaseConnector

log = logging.getLogger(__name__)


class MQTTConnector(BaseConnector):
    def __init__(self, connector_id: str, options, routes, *, on_message):
        super().__init__(connector_id, on_message=on_message)
        self.options = options
        self.routes = routes

    async def iter_messages(self) -> AsyncIterator[EdgeMessage]:
        host = self.options.get("host", "127.0.0.1")
        port = int(self.options.get("port", 1883))
        username = self.options.get("username")
        password = self.options.get("password")
        topics = [route.filter for route in self.routes]
        reconnect_interval = int(self.options.get("reconnect_interval", 5))
        while True:
            try:
                async with Client(hostname=host, port=port, username=username, password=password) as client:
                    log.info("connector %s subscribed to %d topics", self.connector_id, len(topics))
                    async with client.unfiltered_messages() as messages:
                        await client.subscribe([(topic, 0) for topic in topics])
                        async for message in messages:
                            route = self._match_route(message.topic)
                            if route is None:
                                continue
                            msg = EdgeMessage(
                                sensor_id=route.sensor_id or message.topic,
                                payload=message.payload,
                                encoding=route.serializer,
                                metadata={"topic": message.topic},
                                pipeline_override=route.pipeline,
                            )
                            yield msg
            except MqttError:
                log.exception("connector %s lost connection; retrying", self.connector_id)
                await asyncio.sleep(reconnect_interval)

    def _match_route(self, topic: str):
        for route in self.routes:
            if self._topic_matches(route.filter, topic):
                return route
        return None

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        if "#" in pattern_parts:
            idx = pattern_parts.index("#")
            if idx != len(pattern_parts) - 1:
                return False
            pattern_parts = pattern_parts[:idx]
            topic_parts = topic_parts[: len(pattern_parts)]
        if len(pattern_parts) != len(topic_parts):
            return False
        for pp, tp in zip(pattern_parts, topic_parts):
            if pp in {"+", "#"}:
                continue
            if pp != tp:
                return False
        return True
