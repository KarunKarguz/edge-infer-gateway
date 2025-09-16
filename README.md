# edge-infer-gateway

> Ultra-light **TensorRT** inference gateway (C++/TCP, epoll). Serve multiple `.plan` engines over a single port with a tiny binary protocol. Includes Docker image and Python client.

## Why
- Tiny, fast, dependency-light alternative to heavy model servers.
- One process, many models. Works locally or in containers.
- Clean building block for edge apps and demos.

## Features (MVP)
- Multi-model registry (YAML), lazy engine loading
- TCP/epoll server, per-model worker threads & CUDA streams
- Pinned host buffers, preallocated device bindings
- Simple binary protocol; Python client examples
- Metrics in logs (mean/p95/qps every 100 reqs)

## Quickstart
```bash
# 1) clone and enter
git clone https://github.com/<you>/edge-infer-gateway.git
cd edge-infer-gateway

# 2) (option A) build & run via Docker
docker build -t edge-infer-gateway -f docker/Dockerfile .
docker run --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -p 8008:8008 edge-infer-gateway

# 2) (option B) build on host/container
bash tools/download_assets.sh
bash tools/build_engine.sh assets/mobilenetv2.onnx
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/edge-infer-gateway --config config/models.yaml --port 8008

# 3) client test
python3 -m pip install pillow numpy
python3 clients/python/examples/classify_image.py \
  --host 127.0.0.1 --port 8008 \
  --model mobilenet_v2_cls \
  --image assets/tabby_tiger_cat.jpg \
  --labels assets/imagenet_classes.txt
