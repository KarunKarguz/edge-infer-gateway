# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test for orchestrator + MQTT + stub gateway."""
from __future__ import annotations

import asyncio
import json
import socket
import struct
from contextlib import asynccontextmanager, suppress
from typing import Dict

import numpy as np
import pytest
from asyncio_mqtt import Client
from amqtt.broker import Broker
import paho.mqtt.client as mqtt

if not hasattr(mqtt.Client, "message_retry_set"):
    def _message_retry_set(self, _value):  # pragma: no cover - legacy shim
        return None

    mqtt.Client.message_retry_set = _message_retry_set  # type: ignore[attr-defined]

from orchestrator.app import EdgeOrchestrator
from orchestrator.config import load_config
from orchestrator.gateway_pool import InferenceResult


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class StubGateway:
    """Minimal TCP server that mimics the TensorRT gateway protocol for tests."""

    def __init__(self, responses: Dict[str, np.ndarray], host: str = "127.0.0.1", port: int = 0):
        self._responses = responses
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None

    @property
    def port(self) -> int:
        if not self._server:
            raise RuntimeError("gateway not started")
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                length_data = await reader.readexactly(4)
                (frame_len,) = struct.unpack("<I", length_data)
                payload = await reader.readexactly(frame_len)
                magic, version, flags, model_len, tensor_count, _ = struct.unpack_from("<4sHHIII", payload, 0)
                if magic != b"TRT\x01" or version != 1:
                    break
                offset = struct.calcsize("<4sHHIII")
                model_id = payload[offset : offset + model_len].decode()
                offset += model_len
                for _ in range(tensor_count):
                    _, ndim = struct.unpack_from("<BB", payload, offset)
                    offset += 2
                    offset += 4 * ndim  # dims (int32)
                    (raw_len,) = struct.unpack_from("<I", payload, offset)
                    offset += 4 + raw_len
                vector = self._responses.get(model_id, np.zeros((1,), dtype=np.float32))
                blob = np.asarray(vector, dtype=np.float32).tobytes()
                body = struct.pack("<III", 1, 0, 1)
                body += struct.pack("<I", len(blob))
                body += blob
                writer.write(struct.pack("<I", len(body)) + body)
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()


@asynccontextmanager
async def mqtt_broker(host: str, port: int):
    config = {
        "listeners": {"default": {"type": "tcp", "bind": f"{host}:{port}"}},
        "sys_interval": 0,
        "topic-check": {"enabled": False},
    }
    broker = Broker(config)
    await broker.start()
    try:
        yield
    finally:
        await broker.shutdown()


def preprocess_vector(message, payload):
    values = [float(v) for v in payload.values()]
    arr = np.asarray(values, dtype=np.float32)
    return [arr[np.newaxis, :]]


def postprocess_vector(result: InferenceResult, message):
    vector = np.frombuffer(result.outputs[0], dtype=np.float32)
    return {
        "sensor": message.sensor_id,
        "vector": vector.tolist(),
    }


async def _wait_for(predicate, timeout: float = 5.0):
    end_time = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end_time:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met within timeout")


@pytest.mark.asyncio
async def test_orchestrator_end_to_end(tmp_path):
    gateway = StubGateway({"test_model": np.asarray([0.1, 0.2, 0.7], dtype=np.float32)})
    await gateway.start()

    mqtt_port = _free_port()
    async with mqtt_broker("127.0.0.1", mqtt_port):
        config_path = tmp_path / "pipelines.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "gateway:",
                    "  host: 127.0.0.1",
                    f"  port: {gateway.port}",
                    "  pool_size: 1",
                    "  timeout_s: 1.0",
                    "connectors:",
                    "  - id: test-mqtt",
                    "    type: mqtt",
                    "    host: 127.0.0.1",
                    f"    port: {mqtt_port}",
                    "    topics:",
                    "      - filter: tests/env",
                    "        pipeline: test-pipeline",
                    "        serializer: json",
                    "pipelines:",
                    "  - id: test-pipeline",
                    "    preprocess: tests.test_integration:preprocess_vector",
                    "    model: test_model",
                    "    postprocess: tests.test_integration:postprocess_vector",
                    "    agents:",
                    "      - test_capture",
                    "agents:",
                    "  test_capture:",
                    "    type: capture",
                    "actions:",
                    "  log:",
                    "    type: log",
                    "metrics_port: 0",
                ]
            )
        )

        orchestrator = EdgeOrchestrator(load_config(config_path))
        await orchestrator.start()
        try:
            await asyncio.sleep(0.2)
            async with Client(hostname="127.0.0.1", port=mqtt_port, message_retry_set=None) as client:
                payload = json.dumps({"a": 1.0, "b": 2.0, "c": 3.0}).encode("utf-8")
                await client.publish("tests/env", payload)

            capture_agent = orchestrator.agent_registry["test_capture"]
            await _wait_for(lambda: bool(capture_agent.events))
            event = capture_agent.events[0]
            assert event["payload"]["sensor"] == "tests/env"
            assert event["payload"]["vector"] == pytest.approx([0.1, 0.2, 0.7])
            assert event["latency_ms"] < 250
        finally:
            await orchestrator.stop()

    await gateway.stop()
