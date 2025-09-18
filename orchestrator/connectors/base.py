"""Connector primitives for ingesting sensor data."""
from __future__ import annotations

import abc
import asyncio
import contextlib
from typing import AsyncIterator, Callable, Optional

from orchestrator.messages import EdgeMessage


class BaseConnector(abc.ABC):
    def __init__(self, connector_id: str, *, on_message: Callable[[EdgeMessage], asyncio.Future | None]):
        self.connector_id = connector_id
        self._on_message = on_message
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name=f"connector-{self.connector_id}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        async for msg in self.iter_messages():
            await self._on_message(msg)

    @abc.abstractmethod
    def iter_messages(self) -> AsyncIterator[EdgeMessage]:  # pragma: no cover - interface
        raise NotImplementedError
