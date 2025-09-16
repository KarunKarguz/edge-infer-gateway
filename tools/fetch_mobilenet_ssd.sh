#!/usr/bin/env bash
set -euo pipefail
mkdir -p assets
echo "Attempting to fetch SSD MobileNet ONNX models (v1/v2)..."

urls=(
  # Try a few known locations; these may move over time
  "https://github.com/onnx/models/releases/download/ssd-mobilenetv1-12/ssd-mobilenetv1-12.onnx"
  "https://zenodo.org/record/6600531/files/ssd_mobilenet_v1_coco_2018_01_28.onnx?download=1"
)

ok=0
for u in "${urls[@]}"; do
  echo "Trying $u"
  if curl -L --fail -o assets/ssd_mobilenet.onnx "$u"; then
    ok=1; break
  fi
done

if [[ "$ok" -eq 0 ]]; then
  echo "Could not auto-download SSD MobileNet ONNX."
  echo "Please place your ONNX at assets/ssd_mobilenet.onnx and re-run"
  exit 1
fi

echo "Saved to assets/ssd_mobilenet.onnx"

