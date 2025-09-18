# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution primitives."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, List, Sequence

import numpy as np

from .actions import dispatcher
from .agents.base import Agent
from .config import PipelineConfig
from .gateway_pool import GatewayPool, InferenceResult
from .messages import EdgeMessage
from .serialization import decode_payload
from .utils import resolve_callable

log = logging.getLogger(__name__)

PreprocessFn = Callable[[EdgeMessage, object], Iterable[np.ndarray]]
PostprocessFn = Callable[[InferenceResult, EdgeMessage], object]


@dataclass
class Pipeline:
    cfg: PipelineConfig
    preprocess_fn: PreprocessFn
    postprocess_fn: PostprocessFn | None
    agents: List[Agent]
    _semaphore: asyncio.Semaphore | None = None

    def __post_init__(self) -> None:
        if self.cfg.max_parallel:
            self._semaphore = asyncio.Semaphore(self.cfg.max_parallel)

    async def run(self, message: EdgeMessage, gateway: GatewayPool) -> None:
        start = time.perf_counter()
        payload_obj = decode_payload(message)
        arrays = list(self.preprocess_fn(message, payload_obj))
        inference_latency = 0.0
        if self.cfg.model and arrays:
            guard = self._semaphore
            if guard:
                async with guard:
                    result = await gateway.infer(self.cfg.model, arrays)
            else:
                result = await gateway.infer(self.cfg.model, arrays)
            inference_latency = (time.perf_counter() - start) * 1000
            if result.status != 0:
                log.error("pipeline %s inference failed status=%s", self.cfg.id, result.status)
                return
            post_obj = self.postprocess_fn(result, message) if self.postprocess_fn else result
        elif self.cfg.model and not arrays:
            log.warning("pipeline %s received empty tensors from %s", self.cfg.id, message.sensor_id)
            return
        else:
            post_obj = payload_obj
        await self._run_agents(message, post_obj, inference_latency)

    async def _run_agents(self, message: EdgeMessage, data: object, latency_ms: float) -> None:
        for agent in self.agents:
            try:
                actions = await agent.handle(message=message, payload=data, latency_ms=latency_ms)
            except Exception:
                log.exception("agent %s failed", agent.name)
                continue
            for action in actions or []:
                await dispatcher.dispatch(action, agent=agent.name, pipeline=self.cfg.id)


class PipelineFactory:
    def __init__(self, pipeline_cfg: PipelineConfig):
        self.cfg = pipeline_cfg

    def build(self, agent_registry: dict[str, Agent]) -> Pipeline:
        preprocess = resolve_callable(self.cfg.preprocess)
        postprocess = resolve_callable(self.cfg.postprocess) if self.cfg.postprocess else None
        agents = [agent_registry[name] for name in self.cfg.agents]
        return Pipeline(cfg=self.cfg, preprocess_fn=preprocess, postprocess_fn=postprocess, agents=agents)
