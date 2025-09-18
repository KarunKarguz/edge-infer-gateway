"""Configuration loader for orchestrator pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(slots=True)
class GatewayConfig:
    host: str
    port: int
    pool_size: int = 4
    timeout_s: float = 2.0


@dataclass(slots=True)
class TopicRoute:
    filter: str
    pipeline: str
    serializer: str = "json"
    sensor_id: Optional[str] = None


@dataclass(slots=True)
class ConnectorConfig:
    id: str
    type: str
    options: Dict[str, Any] = field(default_factory=dict)
    topics: List[TopicRoute] = field(default_factory=list)


@dataclass(slots=True)
class PipelineConfig:
    id: str
    preprocess: str
    model: Optional[str] = None
    postprocess: Optional[str] = None
    agents: List[str] = field(default_factory=list)
    deadline_ms: Optional[int] = None
    max_parallel: Optional[int] = None


@dataclass(slots=True)
class ActionConfig:
    name: str
    type: str
    options: Dict[str, Any]


@dataclass(slots=True)
class OrchestratorConfig:
    version: int
    gateway: GatewayConfig
    connectors: List[ConnectorConfig]
    pipelines: Dict[str, PipelineConfig]
    actions: List[ActionConfig]
    agents: Dict[str, Dict[str, Any]]
    metrics_port: int = 9108


def _parse_gateway(data: Dict[str, Any]) -> GatewayConfig:
    return GatewayConfig(
        host=data.get("host", "127.0.0.1"),
        port=int(data.get("port", 8008)),
        pool_size=int(data.get("pool_size", 4)),
        timeout_s=float(data.get("timeout_s", 2.0)),
    )


def _parse_connectors(items: List[Dict[str, Any]]) -> List[ConnectorConfig]:
    connectors: List[ConnectorConfig] = []
    for item in items:
        topics = [
            TopicRoute(
                filter=topic.get("filter"),
                pipeline=topic.get("pipeline"),
                serializer=topic.get("serializer", "json"),
                sensor_id=topic.get("sensor_id"),
            )
            for topic in item.get("topics", [])
        ]
        connectors.append(
            ConnectorConfig(
                id=item["id"],
                type=item["type"],
                options={k: v for k, v in item.items() if k not in {"id", "type", "topics"}},
                topics=topics,
            )
        )
    return connectors


def _parse_pipelines(items: List[Dict[str, Any]]) -> Dict[str, PipelineConfig]:
    pipelines: Dict[str, PipelineConfig] = {}
    for item in items:
        cfg = PipelineConfig(
            id=item["id"],
            preprocess=item["preprocess"],
            model=item.get("model"),
            postprocess=item.get("postprocess"),
            agents=item.get("agents", []) or [],
            deadline_ms=item.get("deadline_ms"),
            max_parallel=item.get("max_parallel"),
        )
        pipelines[cfg.id] = cfg
    return pipelines


def _parse_actions(items: List[Dict[str, Any]]) -> List[ActionConfig]:
    actions: List[ActionConfig] = []
    for name, payload in items.items():
        if not isinstance(payload, dict):
            raise ValueError(f"action '{name}' must be a mapping")
        action_type = payload.get("type", name)
        options = {k: v for k, v in payload.items() if k != "type"}
        actions.append(ActionConfig(name=name, type=action_type, options=options))
    return actions


def load_config(path: str | Path) -> OrchestratorConfig:
    raw = yaml.safe_load(Path(path).read_text())
    version = int(raw.get("version", 1))
    gateway = _parse_gateway(raw.get("gateway", {}))
    connectors = _parse_connectors(raw.get("connectors", []))
    pipelines = _parse_pipelines(raw.get("pipelines", []))
    actions = _parse_actions(raw.get("actions", {}))
    agents = raw.get("agents", {})
    metrics_port = int(raw.get("metrics_port", 9108))
    return OrchestratorConfig(
        version=version,
        gateway=gateway,
        connectors=connectors,
        pipelines=pipelines,
        actions=actions,
        agents=agents,
        metrics_port=metrics_port,
    )
