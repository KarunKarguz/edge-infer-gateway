#!/usr/bin/env bash
set -euo pipefail

# Generic ONNX -> TensorRT engine builder
# Usage:
#   tools/build_engine.sh assets/model.onnx models/model.plan [--fp16]

onnx="${1:-assets/mobilenetv2.onnx}"
out="${2:-models/mobilenetv2_fp32.plan}"
shift || true
shift || true
opts=("--onnx=$onnx" "--memPoolSize=workspace:512" "--saveEngine=$out" "--avgRuns=50")
if [[ "${1:-}" == "--fp16" ]]; then
  opts+=("--fp16")
fi
mkdir -p models
echo "Building TensorRT engine: $out from $onnx"
trtexec "${opts[@]}"
echo "Wrote $out"
