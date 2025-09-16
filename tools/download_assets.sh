#!/usr/bin/env bash
set -e
mkdir -p assets
# mobilenet v2 (opset 11) from ONNX Model Zoo (mirror via GitHub raw)
curl -L -o assets/mobilenetv2.onnx https://media.githubusercontent.com/media/onnx/models/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
# sample image & labels (you can swap your own)
curl -L -o assets/tabby_tiger_cat.jpg https://raw.githubusercontent.com/onnx/models/main/vision/classification/resnet/model/tabby_tiger_cat.jpg
curl -L -o assets/imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
