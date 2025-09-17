#!/usr/bin/env bash
set -e
mkdir -p assets

fetch_if_missing(){
  local dest="$1"; shift
  local min_size_bytes="${1:-0}"; shift || true
  local urls=("$@")
  if [[ -f "$dest" ]]; then
    echo "Skipping $(basename "$dest") (already present)"
    return 0
  fi
  for u in "${urls[@]}"; do
    echo "Fetching $(basename "$dest") from $u"
    if curl -L --fail -o "$dest" "$u"; then
      if [[ "$min_size_bytes" -gt 0 ]]; then
        local sz
        sz=$(stat -c%s "$dest")
        if [[ "$sz" -lt "$min_size_bytes" ]]; then
          echo "$(basename "$dest") looks truncated ($sz bytes < $min_size_bytes); retrying alternate source" >&2
          rm -f "$dest"
          continue
        fi
      fi
      return 0
    fi
  done
  echo "WARNING: failed to download $(basename "$dest")" >&2
  return 1
}

# mobilenet v2 (optional classification demo)
if [[ "${EIG_FETCH_MOBILENET:-1}" == "1" ]]; then
  echo "Downloading MobileNetV2 ONNX + sample assets..."
  fetch_if_missing assets/mobilenetv2.onnx 1000000 \
    "https://github.com/onnx/models/releases/download/mobilenetv2-7/mobilenetv2-7.onnx" \
    "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx" \
    "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx" || true
  fetch_if_missing assets/tabby_tiger_cat.jpg 0 \
    "https://raw.githubusercontent.com/onnx/models/main/vision/classification/resnet/model/tabby_tiger_cat.jpg" || true
  fetch_if_missing assets/imagenet_classes.txt 0 \
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" || true
else
  echo "Skipping MobileNetV2 assets (EIG_FETCH_MOBILENET=0)"
fi

echo "Fetching COCO class labels..."
fetch_if_missing assets/coco.names 0 \
  "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names" \
  "https://raw.githubusercontent.com/ultralytics/yolov5/v7.0/data/coco.names" \
  "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

echo "Downloading YOLOv5n/s ONNX (v7.0 release)..."
fetch_if_missing assets/yolov5n.onnx 1000000 \
  "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
fetch_if_missing assets/yolov5s.onnx 1000000 \
  "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx"

echo "Assets ready under ./assets"

if [[ "${EIG_EXPORT_SSD:-1}" == "1" ]]; then
  echo "Downloading SSD-Mobilenet ONNX (v1.0)..."
  python3 -m pip install onnx onnxruntime --user
  python3 -m pip install torch torchvision --user
  python3 tools/convert_ssd_mobilenet.py
else
  echo "Skipping SSD-Mobilenet export (EIG_EXPORT_SSD=0)"  
fi