# SPDX-License-Identifier: Apache-2.0
"""Action primitives used by agents to trigger side effects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class Action:
    dispatcher: str
    target: str | None = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDispatcher:
    def __init__(self, name: str, options: Dict[str, Any]):
        self.name = name
        self.options = options

    async def dispatch(self, action: Action, *, agent: str, pipeline: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError
