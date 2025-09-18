"""Connection pool for the TensorRT TCP gateway."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from clients.python.gateway_stream import GatewayStream

log = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    status: int
    outputs: Sequence[bytes]


class GatewayPool:
    def __init__(self, host: str, port: int, pool_size: int = 4, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: asyncio.Queue[GatewayStream] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            for _ in range(self.pool_size):
                stream = await asyncio.to_thread(GatewayStream, self.host, self.port, self.timeout)
                self._pool.put_nowait(stream)
            self._started = True
            log.info("gateway pool primed with %d connections", self.pool_size)

    async def close(self) -> None:
        while not self._pool.empty():
            stream = self._pool.get_nowait()
            try:
                await asyncio.to_thread(stream.close)
            finally:
                self._pool.task_done()
        self._started = False

    async def infer(self, model_id: str, arrays: Iterable[np.ndarray]) -> InferenceResult:
        if not self._started:
            await self.start()
        stream = await self._pool.get()
        requeue = True
        try:
            status, outputs = await asyncio.to_thread(stream.infer, model_id, list(arrays))
            return InferenceResult(status=status, outputs=outputs)
        except Exception:
            log.exception("inference failed; dropping socket and recreating")
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.close)
            replacement = await asyncio.to_thread(GatewayStream, self.host, self.port, self.timeout)
            await self._pool.put(replacement)
            requeue = False
            raise
        finally:
            if requeue:
                with contextlib.suppress(asyncio.QueueFull):
                    self._pool.put_nowait(stream)
