# SPDX-License-Identifier: Apache-2.0
"""Camera connector reading frames via OpenCV."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import cv2

from orchestrator.messages import EdgeMessage

from .base import BaseConnector

log = logging.getLogger(__name__)


class CameraConnector(BaseConnector):
    def __init__(self, connector_id: str, options, *, on_message):
        super().__init__(connector_id, on_message=on_message)
        self.options = options

    async def iter_messages(self) -> AsyncIterator[EdgeMessage]:
        source = self.options.get("source", 0)
        interval = float(self.options.get("interval", 0.1))
        encoding = self.options.get("encoding", "bgr")
        sensor_id = self.options.get("sensor_id", f"camera:{source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"camera source {source} could not be opened")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("connector %s failed to read frame", self.connector_id)
                    await asyncio.sleep(interval)
                    continue
                payload = frame.tobytes()
                msg = EdgeMessage(
                    sensor_id=sensor_id,
                    payload=payload,
                    encoding=encoding,
                    metadata={"shape": frame.shape},
                    pipeline_override=self.options.get("pipeline"),
                )
                yield msg
                await asyncio.sleep(interval)
        finally:
            cap.release()
