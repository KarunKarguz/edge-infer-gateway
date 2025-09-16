#!/usr/bin/env bash
set -euo pipefail
onnx="${1:-assets/mobilenetv2.onnx}"
out="models/mobilenetv2_fp32.plan"
mkdir -p models
trtexec --onnx="$onnx" --memPoolSize=workspace:512 --saveEngine="$out" --avgRuns=50
echo "Wrote $out"
