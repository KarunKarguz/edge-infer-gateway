#!/usr/bin/env bash
set -euo pipefail

if [[ "${EIG_AUTO_ASSET_DOWNLOAD:-1}" == "1" ]]; then
  mkdir -p assets
  echo "[entrypoint] Ensuring required assets are present"
  if ! bash tools/download_assets.sh; then
    echo "[entrypoint] Warning: asset download reported issues" >&2
  fi
fi

if [[ "${EIG_BUILD_ENGINES:-1}" == "1" ]]; then
  mkdir -p models

  build_engine(){
    local onnx="$1"; local plan="$2"; shift 2
    local args=("$@")
    if [[ ! -f "$onnx" ]]; then
      echo "[entrypoint] Skipping $(basename "$plan") (missing $onnx)"
      return
    fi
    if [[ "${EIG_REUSE_ENGINES:-0}" == "1" && -f "$plan" ]]; then
      echo "[entrypoint] Reusing existing $(basename "$plan")"
      return
    fi
    echo "[entrypoint] Building $(basename "$plan")"
    if ! bash tools/build_engine.sh "$onnx" "$plan" "${args[@]}"; then
      echo "[entrypoint] build failed for $(basename "$plan")" >&2
    fi
  }

  build_engine assets/mobilenetv2.onnx models/mobilenetv2_fp32.plan
  build_engine assets/yolov5n.onnx     models/yolov5n_fp16.plan     --fp16
  build_engine assets/yolov5s.onnx     models/yolov5s_fp16.plan     --fp16

  if [[ "${EIG_EXPORT_SSD:-1}" == "1" ]]; then
    build_engine assets/ssd_mobilenet.onnx models/ssd_mobilenet_fp16.plan --fp16
  fi  
  
fi

exec "$@"
