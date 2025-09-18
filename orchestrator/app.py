"""Async orchestrator runner."""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import Dict, Tuple

from prometheus_client import start_http_server

from orchestrator import agents
from orchestrator.actions import dispatcher as action_dispatcher
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.connectors import create_connector
from orchestrator.gateway_pool import GatewayPool
from orchestrator.metrics import PIPELINE_DROPPED, PIPELINE_INGRESS, PIPELINE_LATENCY, QUEUE_DEPTH
from orchestrator.pipeline import PipelineFactory
from orchestrator.messages import EdgeMessage
from orchestrator.agents.base import Agent

log = logging.getLogger("orchestrator")


class EdgeOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.gateway = GatewayPool(
            host=config.gateway.host,
            port=config.gateway.port,
            pool_size=config.gateway.pool_size,
            timeout=config.gateway.timeout_s,
        )
        self.queue: asyncio.Queue[Tuple[str | None, EdgeMessage | None]] = asyncio.Queue(maxsize=1024)
        self.pipelines = {}
        self.connectors = []
        self.agent_registry: Dict[str, Agent] = {}
        self._workers: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        action_dispatcher.initialise(self.config.actions)
        self.agent_registry = agents.build_agents(self.config.agents)
        for agent in self.agent_registry.values():
            await agent.start()
        for p_cfg in self.config.pipelines.values():
            factory = PipelineFactory(p_cfg)
            self.pipelines[p_cfg.id] = factory.build(self.agent_registry)
        for conn_cfg in self.config.connectors:
            connector = create_connector(conn_cfg, on_message=self._handle_message)
            self.connectors.append(connector)
            await connector.start()
        worker_count = max(2, len(self.pipelines))
        for idx in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(idx), name=f"worker-{idx}"))
        start_http_server(self.config.metrics_port)
        log.info("orchestrator started with %d pipelines, %d connectors", len(self.pipelines), len(self.connectors))

    async def stop(self) -> None:
        self._stop_event.set()
        for _ in self._workers:
            await self.queue.put((None, None))
        for connector in self.connectors:
            await connector.stop()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        for agent in self.agent_registry.values():
            await agent.stop()
        await self.gateway.close()
        await action_dispatcher.close()

    async def _handle_message(self, message) -> None:
        pipeline_id = message.pipeline_override
        if not pipeline_id:
            log.warning("message from %s missing pipeline mapping", message.sensor_id)
            PIPELINE_DROPPED.labels("unknown", "unmapped").inc()
            return
        if pipeline_id not in self.pipelines:
            log.warning("pipeline %s not registered", pipeline_id)
            PIPELINE_DROPPED.labels(pipeline_id, "unregistered").inc()
            return
        try:
            self.queue.put_nowait((pipeline_id, message))
            QUEUE_DEPTH.set(self.queue.qsize())
            PIPELINE_INGRESS.labels(pipeline_id).inc()
        except asyncio.QueueFull:
            PIPELINE_DROPPED.labels(pipeline_id, "queue_full").inc()
            log.error("pipeline %s queue full; dropping message", pipeline_id)

    async def _worker_loop(self, idx: int) -> None:
        while not self._stop_event.is_set():
            pipeline_id, message = await self.queue.get()
            if pipeline_id is None or message is None:
                self.queue.task_done()
                break
            QUEUE_DEPTH.set(self.queue.qsize())
            pipeline = self.pipelines[pipeline_id]
            if pipeline.cfg.deadline_ms:
                age = _latency_ms(message.timestamp)
                if age > pipeline.cfg.deadline_ms:
                    PIPELINE_DROPPED.labels(pipeline_id, "deadline").inc()
                    log.warning(
                        "pipeline %s dropping message (age %.2fms > deadline %sms)",
                        pipeline_id,
                        age,
                        pipeline.cfg.deadline_ms,
                    )
                    self.queue.task_done()
                    continue
            try:
                await pipeline.run(message, self.gateway)
                latency = _latency_ms(message.timestamp)
                PIPELINE_LATENCY.labels(pipeline_id).observe(latency)
            except Exception:
                PIPELINE_DROPPED.labels(pipeline_id, "exception").inc()
                log.exception("pipeline %s processing failed", pipeline_id)
            finally:
                self.queue.task_done()


def _latency_ms(timestamp: datetime) -> float:
    now = datetime.now(timezone.utc)
    delta = now - timestamp
    return delta.total_seconds() * 1000


async def main_async(args) -> None:
    config = load_config(args.config)
    orchestrator = EdgeOrchestrator(config)
    await orchestrator.start()
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows fallback
            pass

    await stop_event.wait()
    log.info("shutdown requested")
    await orchestrator.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Edge orchestrator for TensorRT gateway")
    parser.add_argument("--config", default="config/pipelines.yaml")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
