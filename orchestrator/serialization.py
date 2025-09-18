# SPDX-License-Identifier: Apache-2.0
"""Helpers for parsing connector payloads."""
from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np

from .messages import EdgeMessage


def decode_payload(message: EdgeMessage) -> Any:
    fmt = message.encoding.lower()
    if fmt == "json":
        return json.loads(message.payload.decode("utf-8"))
    if fmt in {"jpg", "jpeg", "image/jpeg"}:
        return message.payload
    if fmt == "base64":
        return base64.b64decode(message.payload)
    if fmt == "npz":
        with np.load(io.BytesIO(message.payload), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    return message.payload
