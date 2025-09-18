"""Connector factory."""
from __future__ import annotations

from typing import Callable

from orchestrator.config import ConnectorConfig

from .base import BaseConnector


CONNECTOR_TYPES: dict[str, Callable[..., BaseConnector]] = {}


def register(conn_type: str, factory: Callable[..., BaseConnector]) -> None:
    CONNECTOR_TYPES[conn_type] = factory


def create_connector(cfg: ConnectorConfig, *, on_message) -> BaseConnector:
    if cfg.type not in CONNECTOR_TYPES:
        raise ValueError(f"unknown connector type '{cfg.type}'")
    return CONNECTOR_TYPES[cfg.type](cfg.id, cfg.options, routes=cfg.topics, on_message=on_message)


from .mqtt import MQTTConnector
from .camera import CameraConnector
from .ble import BLEConnector

register("mqtt", lambda connector_id, options, routes, on_message: MQTTConnector(connector_id, options, routes, on_message=on_message))
register("camera", lambda connector_id, options, routes, on_message: CameraConnector(connector_id, options, on_message=on_message))
register("ble", lambda connector_id, options, routes, on_message: BLEConnector(connector_id, options, on_message=on_message))
