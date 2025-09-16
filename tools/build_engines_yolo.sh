#!/usr/bin/env bash
set -euo pipefail

# Build YOLOv5n/s FP16 engines (faster on most GPUs)
mkdir -p models
if [[ ! -f assets/yolov5n.onnx || ! -f assets/yolov5s.onnx ]]; then
  echo "YOLOv5 ONNX not found under assets/. Run tools/download_assets.sh first." >&2
  exit 1
fi

bash "$(dirname "$0")/build_engine.sh" assets/yolov5n.onnx models/yolov5n_fp16.plan --fp16
bash "$(dirname "$0")/build_engine.sh" assets/yolov5s.onnx models/yolov5s_fp16.plan --fp16
echo "Built: models/yolov5n_fp16.plan, models/yolov5s_fp16.plan"

