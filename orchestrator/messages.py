"""Message envelope shared across connectors, pipelines, and agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(slots=True)
class EdgeMessage:
    """Canonical wrapper around upstream sensor payloads."""

    sensor_id: str
    payload: bytes
    encoding: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_override: Optional[str] = None

    def with_pipeline(self, pipeline_id: str) -> "EdgeMessage":
        msg = EdgeMessage(
            sensor_id=self.sensor_id,
            payload=self.payload,
            encoding=self.encoding,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
            pipeline_override=pipeline_id,
        )
        return msg


def ensure_bytes(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    raise TypeError(f"cannot convert {type(data)} to bytes")
