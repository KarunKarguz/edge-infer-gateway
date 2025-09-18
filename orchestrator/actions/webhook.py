"""HTTP webhook dispatcher for enterprise integrations."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import aiohttp

from .base import Action, BaseDispatcher

log = logging.getLogger(__name__)


class WebhookDispatcher(BaseDispatcher):
    def __init__(self, name: str, options: Dict[str, Any]):
        super().__init__(name, options)
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is not None:
                return self._session
            timeout = aiohttp.ClientTimeout(total=self.options.get("timeout", 5))
            self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    async def dispatch(self, action: Action, *, agent: str, pipeline: str) -> None:
        session = await self._ensure()
        url = action.target or self.options.get("url")
        if not url:
            log.warning("webhook dispatcher %s missing url", self.name)
            return
        method = self.options.get("method", "POST").upper()
        headers = {**self.options.get("headers", {}), "X-Agent": agent, "X-Pipeline": pipeline}
        json_payload = {"agent": agent, "pipeline": pipeline, **action.payload}
        async with session.request(method, url, json=json_payload, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                log.error("webhook %s failed status=%s body=%s", self.name, resp.status, body[:200])

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
