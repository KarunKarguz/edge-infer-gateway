import torch
import torchvision
from pathlib import Path

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

dummy = torch.randn(1, 3, 320, 320)
torch.onnx.export(
    model, dummy, "assets/ssd_mobilenet.onnx",
    input_names=["images"], output_names=["detections"],
    opset_version=13, dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}}
)
