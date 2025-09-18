#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish synthetic sensor payloads to MQTT for orchestrator testing."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from typing import Iterable

from asyncio_mqtt import Client


async def publish_env(client: Client, topic: str, interval: float) -> None:
    t = 0.0
    while True:
        co2 = 500 + 300 * math.sin(t)
        temp = 22 + 1.5 * math.sin(t / 5)
        payload = {
            "co2_ppm": round(co2 + random.uniform(-20, 20), 2),
            "temperature_c": round(temp + random.uniform(-0.5, 0.5), 2),
            "humidity_pct": round(45 + random.uniform(-5, 5), 2),
        }
        await client.publish(topic, json.dumps(payload).encode("utf-8"))
        await asyncio.sleep(interval)
        t += 0.2


async def publish_trigger(client: Client, topic: str, interval: float) -> None:
    while True:
        payload = {
            "label": "person",
            "bbox": [random.randint(0, 100), random.randint(0, 100), random.randint(101, 200), random.randint(101, 200)],
        }
        await client.publish(topic, json.dumps(payload).encode("utf-8"))
        await asyncio.sleep(interval)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("broker")
    ap.add_argument("topic")
    ap.add_argument("--mode", choices=["env", "trigger"], default="env")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    async with Client(hostname=args.broker, port=args.port, username=args.username, password=args.password) as client:
        if args.mode == "env":
            await publish_env(client, args.topic, args.interval)
        else:
            await publish_trigger(client, args.topic, args.interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
