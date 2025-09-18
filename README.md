# edge-infer-gateway

> edge-infer-gateway — a tiny, production-ready TCP gateway for TensorRT engines. Multi-model, concurrency-safe, epoll-driven, health/metrics out of the box. Drop in your .plan and go.

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
- **NEW:** Async edge orchestrator (Python) for multi-sensor ingestion, agent routing, and action dispatch (MQTT/webhook/log)

## New: YOLOv5n/s Object Detection
- Assets downloader now fetches `yolov5n.onnx` and `yolov5s.onnx` from Ultralytics.
- Scripts provided to build FP16 TensorRT engines.
- Config includes `yolov5n_coco` and `yolov5s_coco` model IDs.
- Python examples for single-image detection and live streaming.

## Switch Between Models Easily
- Choose the `--model` id when calling clients:
  - Classification: `mobilenet_v2_cls`
  - Detection (YOLO): `yolov5n_coco` or `yolov5s_coco`
  - Detection (SSD): `ssd_mobilenet_coco` (if you built the engine)
- Clients select preprocessing accordingly:
  - C++: `eig-stream --mode yolo|ssd --model <id>`
  - Python: `detect_stream_yolov5.py` (YOLO) or implement SSD similarly
- C++ streaming client (`eig-stream`) for low-latency webcam/video pipelines.

## Quickstart
```bash
# clone 
git clone https://github.com/<you>/edge-infer-gateway.git
cd edge-infer-gateway

# build & run (Docker build)
docker build -t edge-infer-gateway -f docker/Dockerfile .
docker run --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -p 8008:8008 -p 8080:8080 -p 9108:9108 \
  edge-infer-gateway
# entrypoint launches the TensorRT gateway and the orchestrator. Disable orchestration via EIG_ENABLE_ORCHESTRATOR=0.

# orchestrator only (host GPU)
python3 -m pip install -r requirements.orchestrator.txt
python3 -m orchestrator.app --config config/pipelines.yaml

# build & run locally
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEIG_BUILD_CLIENT_CPP=ON
cmake --build build -j
./build/eig-client --host 127.0.0.1 --port 8008 --model mobilenet_v2_cls --image path/to.jpg


# health
curl localhost:8080/healthz
curl localhost:8080/metrics

# client test
python3 -m pip install numpy pillow
python3 clients/python/examples/classify_image.py \
  --host 127.0.0.1 --port 8008 \
  --model mobilenet_v2_cls \
  --image assets/tabby_tiger_cat.jpg \
  --labels assets/imagenet_classes.txt

## Object Detection Quickstart (YOLOv5)
```bash
# fetch assets (skips existing files)
bash tools/download_assets.sh
# optional classification engine
bash tools/build_engine.sh assets/mobilenetv2.onnx models/mobilenetv2_fp32.plan
# yolo engines (fp16 for speed)
bash tools/build_engines_yolo.sh

# run gateway
./build/edge-infer-gateway --config config/models.yaml --port 8008 --http-port 8080

# single-image detection
python3 -m pip install numpy opencv-python
python3 clients/python/examples/detect_image_yolov5.py \
  --host 127.0.0.1 --port 8008 \
  --model yolov5n_coco \
  --image assets/tabby_tiger_cat.jpg \
  --labels assets/coco.names

# streaming (webcam 0), press ESC to quit
python3 clients/python/examples/detect_stream_yolov5.py --show --source 0 --model yolov5n_coco

# C++ streaming (YOLO default)
./build/eig-stream --host 127.0.0.1 --port 8008 --model yolov5n_coco --mode yolo --source 0

# C++ streaming (SSD)
./build/eig-stream --host 127.0.0.1 --port 8008 --model ssd_mobilenet_coco --mode ssd --source 0

# streaming with GStreamer source (example)
python3 clients/python/examples/detect_stream_yolov5.py \
  --source "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR ! appsink" \
  --model yolov5n_coco --show
```

Notes:
- YOLOv5 engines here are built as FP16; clients send `np.float16` inputs for optimal performance.
- NVIDIA GeForce 920MX (Maxwell) can work with older CUDA/TensorRT; ensure your local environment’s TensorRT version supports your GPU. Inside the provided container (TensorRT 22.12), very old GPUs may not be recognized. For laptops, consider building natively with a matching TensorRT version.
- Classification demo downloads can be skipped by setting `EIG_FETCH_MOBILENET=0` before running `tools/download_assets.sh` (the Dockerfile already honors this via env override).
- Edge orchestrator design + pipeline configuration live in `docs/SYSTEM_DESIGN.md`.

## Host-Mounted Assets & Engines
- Persist ONNX assets and TensorRT plans by binding host directories:
  ```bash
  mkdir -p assets models
  sudo chown -R 10001:10001 assets models
  docker run --gpus all \
    -v $(pwd)/assets:/app/assets \
    -v $(pwd)/models:/app/models \
    -p 8008:8008 -p 8080:8080 \
    edge-infer-gateway
  ```
- Engines rebuild on each container start by default to capture driver/architecture changes. Set `-e EIG_REUSE_ENGINES=1` to reuse existing plans without rebuilding.
- Override `-e EIG_FETCH_MOBILENET=0` to skip classification assets (already default in the Dockerfile).
- Asset download runs automatically on startup (`EIG_AUTO_ASSET_DOWNLOAD=1`). Disable if you manage ONNX files manually.

## Edge Orchestrator
- Configure pipelines, agents, and action dispatchers in `config/pipelines.yaml` (see `docs/SYSTEM_DESIGN.md`).
- Runtime knobs:
  - `EIG_ENABLE_ORCHESTRATOR=0` disables the orchestrator (gateway only).
  - `EIG_PIPELINES_CONFIG` overrides the orchestrator config path.
- Prometheus metrics listen on the port declared in `config.metrics_port` (defaults to 9108).
- MQTT/BLE/OpenCV dependencies are preinstalled in the container via `requirements.orchestrator.txt`. Install the same file on bare-metal deployments.

## Tests
- Dev dependencies: `python3 -m pip install -r requirements.dev.txt`
- Run orchestrator integration smoke test: `pytest tests/test_integration.py`
- The integration harness spins up a stub TensorRT gateway, an in-process MQTT broker, and the Python orchestrator to validate model routing + latency reporting.
- See `CONTRIBUTING.md` for full contributor workflow and expectations.

### C++ Streaming Client
Build with OpenCV enabled to use the streaming client:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEIG_BUILD_CLIENT_CPP=ON
cmake --build build -j

# YOLOv5 (expects FP16 inputs) from webcam index 0
./build/eig-stream --host 127.0.0.1 --port 8008 --model yolov5n_coco --source 0

# SSD MobileNet (expects FP32 1x3x300x300)
./build/eig-stream --host 127.0.0.1 --port 8008 --model ssd_mobilenet_coco --source 0
```

### Python Streaming for SSD
```bash
python3 clients/python/examples/detect_stream_ssd.py --source 0 --model ssd_mobilenet_coco --show
```

### Benchmark
```bash
python3 clients/python/examples/benchmark.py --model yolov5n_coco --iters 200
python3 clients/python/examples/benchmark.py --model mobilenet_v2_cls --iters 200
```
Outputs include throughput plus mean/p50/p95/p99 latency (milliseconds).

## Latency & Debugging
- Server emits JSON log lines (`infer_ok ms=…`), making latency scraping trivial with jq/Fluent Bit.
- `clients/python/examples/benchmark.py` provides quick throughput + percentile measurements; integrate it into CI for smoke perf tests.
- Inspect `docs/ARCHITECTURE.md` for an architectural overview, deployment tips, and profiling pointers (Nsight Systems, `trtexec --loadEngine ... --dumpProfile`).

## Optional: SSD MobileNet
- Try fetching an SSD MobileNet ONNX and building a TensorRT engine:
```bash
bash tools/fetch_mobilenet_ssd.sh || true  # places assets/ssd_mobilenet.onnx if successful
# build (may fail if ONNX uses unsupported ops for your TRT version)
bash tools/build_engine.sh assets/ssd_mobilenet.onnx models/ssd_mobilenet_fp32.plan
# then use model id: ssd_mobilenet_coco
```

## Streaming tips (FFmpeg / GStreamer)
- GStreamer webcam source string example:
  - `v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR ! appsink`
- FFmpeg to RTSP/UDP is possible by piping annotated frames out of Python via `subprocess.Popen([ffmpeg ... -f rawvideo -pix_fmt bgr24 -s WxH -i - ...])`. For simplicity, the provided example displays to a window; adapt per your pipeline.

## Jetson (aarch64) Notes
- Use `docker/Dockerfile.jetson` with L4T TensorRT base:
  - `docker build -f docker/Dockerfile.jetson -t eig-jetson .`
  - Run with `--runtime nvidia` and matching JetPack version.
- Or build natively on the device to match the installed JetPack TensorRT:
  - `sudo apt-get install -y build-essential cmake libyaml-cpp-dev`
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
  - `bash tools/download_assets.sh` (sets up YOLO assets; Mobilenet is optional)
  - `bash tools/build_engines_yolo.sh`
- Keep FP16 for YOLO on Jetson; prefer the `yolov5n` engine for lower latency on lower-SM parts.
- Docker entrypoint auto-builds engines at container start. Use `EIG_REUSE_ENGINES=1` when prebuilt plans are mounted.
