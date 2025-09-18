"""BLE connector for Arduino Nano 33 BLE Sense using bleak."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from bleak import BleakClient, BleakScanner

from orchestrator.messages import EdgeMessage

from .base import BaseConnector

log = logging.getLogger(__name__)


class BLEConnector(BaseConnector):
    def __init__(self, connector_id: str, options, *, on_message):
        super().__init__(connector_id, on_message=on_message)
        self.options = options

    async def iter_messages(self) -> AsyncIterator[EdgeMessage]:
        device_name = self.options.get("name")
        service_uuid = self.options.get("service_uuid")
        characteristic_uuid = self.options.get("characteristic_uuid")
        poll_interval = float(self.options.get("poll_interval", 5.0))
        if not (service_uuid and characteristic_uuid):
            raise ValueError("BLE connector requires service_uuid and characteristic_uuid")
        while True:
            device = await BleakScanner.find_device_by_filter(
                lambda d, ad: device_name in d.name if device_name else True
            )
            if device is None:
                log.warning("BLE device %s not found", device_name or service_uuid)
                await asyncio.sleep(poll_interval)
                continue
            try:
                async with BleakClient(device) as client:
                    sensor_id = self.options.get("sensor_id", device.address)
                    while True:
                        data = await client.read_gatt_char(characteristic_uuid)
                        msg = EdgeMessage(
                            sensor_id=sensor_id,
                            payload=data,
                            encoding=self.options.get("encoding", "json"),
                            metadata={"service_uuid": service_uuid},
                            pipeline_override=self.options.get("pipeline"),
                        )
                        yield msg
                        await asyncio.sleep(poll_interval)
            except Exception:
                log.exception("BLE connector %s error; reconnecting", self.connector_id)
                await asyncio.sleep(poll_interval)
