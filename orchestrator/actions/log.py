# SPDX-License-Identifier: Apache-2.0
"""Simple logging dispatcher for debugging and audits."""
from __future__ import annotations

import logging

from .base import Action, BaseDispatcher

log = logging.getLogger(__name__)


class LogDispatcher(BaseDispatcher):
    async def dispatch(self, action: Action, *, agent: str, pipeline: str) -> None:
        log.info("[action %s] %s -> %s payload=%s metadata=%s", pipeline, agent, action.target, action.payload, action.metadata)
