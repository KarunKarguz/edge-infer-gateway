# SPDX-License-Identifier: Apache-2.0
"""Pre/post processing helpers for environmental sensors."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from orchestrator.gateway_pool import InferenceResult
from orchestrator.messages import EdgeMessage


def vector_to_tensor(message: EdgeMessage, payload) -> Iterable[np.ndarray]:
    if isinstance(payload, dict):
        values = [float(payload[k]) for k in sorted(payload.keys()) if isinstance(payload[k], (int, float))]
    elif isinstance(payload, (list, tuple)):
        values = [float(v) for v in payload]
    else:
        raise TypeError("environment payload must be dict or list of numbers")
    arr = np.asarray(values, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    yield arr


def softmax_topk(result: InferenceResult, message: EdgeMessage, k: int = 3) -> List[dict]:
    import numpy as np

    logits = np.frombuffer(result.outputs[0], dtype=np.float32)
    logits = logits.reshape(1, -1)[0]
    exps = np.exp(logits - np.max(logits))
    probs = exps / exps.sum()
    idx = probs.argsort()[::-1][:k]
    return [{"index": int(i), "confidence": float(probs[i])} for i in idx]
