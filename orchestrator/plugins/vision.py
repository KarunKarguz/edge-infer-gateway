# SPDX-License-Identifier: Apache-2.0
"""Vision preprocessing and postprocessing helpers."""
from __future__ import annotations

import io
from typing import Iterable, List

import cv2
import numpy as np

from orchestrator.gateway_pool import InferenceResult
from orchestrator.messages import EdgeMessage


def jpeg_to_yolov5(message: EdgeMessage, payload) -> Iterable[np.ndarray]:
    if isinstance(payload, (bytes, bytearray)):
        data = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        raise TypeError("JPEG payload expected")
    message.metadata["image_hw"] = img.shape[:2]
    img, params = _letterbox(img, 640)
    message.metadata["letterbox"] = params
    arr = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    arr = np.ascontiguousarray(arr)[None, ...]
    yield arr.astype(np.float16)


def bgr_frame_to_yolov5(message: EdgeMessage, payload) -> Iterable[np.ndarray]:
    shape = message.metadata.get("shape")
    if shape is None:
        raise ValueError("camera frame shape missing")
    bgr = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
    message.metadata["image_hw"] = bgr.shape[:2]
    img, params = _letterbox(bgr, 640)
    message.metadata["letterbox"] = params
    arr = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    yield arr.astype(np.float16)[None, ...]


def yolo_nms(result: InferenceResult, message: EdgeMessage, conf_th: float = 0.25, iou_th: float = 0.45) -> dict:
    preds = np.frombuffer(result.outputs[0], dtype=np.float16).astype(np.float32).reshape(1, -1, 85)[0]
    boxes = preds[:, :4]
    scores_obj = _sigmoid(preds[:, 4])
    cls_logits = preds[:, 5:]
    cls_scores = _sigmoid(cls_logits)
    cls_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores.max(axis=1)
    conf = scores_obj * cls_conf
    mask = conf >= conf_th
    boxes = boxes[mask]
    conf = conf[mask]
    cls_ids = cls_ids[mask]
    if boxes.size == 0:
        return []
    xyxy = _xywh_to_xyxy(boxes)
    keep = _nms(xyxy, conf, iou_th)
    ih, iw = message.metadata.get("image_hw", (640, 640))
    gain, pad_w, pad_h, _ = message.metadata.get("letterbox", (1.0, 0.0, 0.0, (ih, iw)))
    detections = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        x1 = (x1 - pad_w) / gain
        x2 = (x2 - pad_w) / gain
        y1 = (y1 - pad_h) / gain
        y2 = (y2 - pad_h) / gain
        x1 = max(0.0, min(x1, iw))
        x2 = max(0.0, min(x2, iw))
        y1 = max(0.0, min(y1, ih))
        y2 = max(0.0, min(y2, ih))
        detections.append({
            "label": int(cls_ids[i]),
            "confidence": float(conf[i]),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        })
    image_blob = message.payload if message.encoding.lower() in {"jpeg", "jpg", "image/jpeg"} else None
    return {
        "detections": detections,
        "image": image_blob,
        "encoding": message.encoding,
        "sensor": message.sensor_id,
    }


def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, (r, dw, dh, shape)


def _xywh_to_xyxy(boxes):
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def _nms(boxes, scores, iou_th):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        idxs = idxs[1:][
            _iou_xyxy(boxes[i], boxes[idxs[1:]]) < iou_th
        ]
    return keep


def _iou_xyxy(b0, B):
    inter_x1 = np.maximum(b0[0], B[:, 0])
    inter_y1 = np.maximum(b0[1], B[:, 1])
    inter_x2 = np.minimum(b0[2], B[:, 2])
    inter_y2 = np.minimum(b0[3], B[:, 3])
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area0 = (b0[2] - b0[0]) * (b0[3] - b0[1])
    areaB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (area0 + areaB - inter + 1e-6)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
