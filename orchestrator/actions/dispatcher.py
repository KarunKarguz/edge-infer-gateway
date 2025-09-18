"""Dispatcher registry that fans out actions to concrete transports."""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from .base import Action, BaseDispatcher
from .log import LogDispatcher
from .mqtt import MQTTDispatcher
from .webhook import WebhookDispatcher

log = logging.getLogger(__name__)

_DISPATCHERS: Dict[str, BaseDispatcher] = {}


def initialise(action_configs) -> None:
    _DISPATCHERS.clear()
    for cfg in action_configs:
        if cfg.type == "log":
            dispatcher = LogDispatcher(cfg.name, cfg.options)
        elif cfg.type == "mqtt":
            dispatcher = MQTTDispatcher(cfg.name, cfg.options)
        elif cfg.type == "webhook":
            dispatcher = WebhookDispatcher(cfg.name, cfg.options)
        else:
            raise ValueError(f"unsupported dispatcher type '{cfg.type}'")
        _DISPATCHERS[cfg.name] = dispatcher
    log.info("registered %d action dispatchers", len(_DISPATCHERS))


async def dispatch(action: Action, *, agent: str, pipeline: str) -> None:
    if action.dispatcher not in _DISPATCHERS:
        log.warning("no dispatcher registered for action %s", action.dispatcher)
        return
    await _DISPATCHERS[action.dispatcher].dispatch(action, agent=agent, pipeline=pipeline)


async def close() -> None:
    await asyncio.gather(*(d.close() for d in _DISPATCHERS.values() if hasattr(d, "close")))
