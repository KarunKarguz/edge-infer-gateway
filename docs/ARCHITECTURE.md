<!-- SPDX-License-Identifier: Apache-2.0 -->

# edge-infer-gateway Architecture

This document captures the layout of the project, how the runtime is structured, and how to extend it for new models or deployment environments. For the full end-to-end system (sensor connectors, agent framework, action dispatch) see `docs/SYSTEM_DESIGN.md`.

## Layout Overview

```
edge-infer-gateway/
├── config/              # YAML configuration for server + models
├── docs/                # Additional documentation
├── include/             # Public headers shared across server components
├── src/                 # Runtime implementation
├── clients/             # Sample clients (Python + C++)
├── tools/               # Utility scripts for assets + engine builds
├── docker/              # Container build + entrypoint scripts
└── assets/, models/     # (Optional) host-mounted inputs + TensorRT plans
```

Key scripts:
- `tools/download_assets.sh` fetches ONNX models and sample assets. Safe to rerun; it skips downloads when files already exist.
- `tools/build_engine.sh` wraps `trtexec` and works with any ONNX model.
- `tools/build_engines_yolo.sh` builds the shipped YOLOv5n/s engines.
- `docker/entrypoint.sh` optionally refreshes assets (`EIG_AUTO_ASSET_DOWNLOAD` requires bind-mounted directories with write access) and builds TensorRT engines unless `EIG_REUSE_ENGINES=1`.

## Runtime Components

### `TRTRunner`
- Owns a TensorRT engine, device bindings, and a pool of execution contexts + CUDA streams.
- Provides `infer(host_inputs, host_outputs)` which handles H2D/D2H copies and context checkout.
- One instance per model (lazy-loaded via `ModelManager`).

### `ModelManager`
- Reads `config/models.yaml` to learn about configured models.
- Lazily constructs a `TRTRunner` the first time a model ID is requested.
- Allows multiple concurrent models in a single server instance.

### `Gateway`
- TCP/epoll loop that multiplexes multiple client connections.
- Each request is read into memory, parsed according to `protocol.hpp`, dispatched to the relevant `TRTRunner`, and responded with raw tensor bytes.
- Simple HTTP sidecar provides `/healthz`, `/readyz`, and `/metrics` counters.
- Per-request latency is logged in JSON (`infer_ok ms=...`).

## Adding a Model

1. Place the ONNX file under `assets/` (or mount it at runtime).
2. Build a TensorRT engine: `bash tools/build_engine.sh assets/<name>.onnx models/<name>.plan [--fp16]`.
3. Update `config/models.yaml` with a new `id`, `engine` path, and metadata.
4. Restart the gateway. On first request, the engine loads and serves.

## Deployment Notes

- Host-mount `assets/` and `models/` when running in containers to avoid re-downloading or re-building on each boot:
  ```bash
  docker run --gpus all \
    -v $(pwd)/assets:/app/assets \
    -v $(pwd)/models:/app/models \
    -p 8008:8008 -p 8080:8080 \
    edge-infer-gateway
  ```
- Set `EIG_REUSE_ENGINES=1` to skip engine rebuilds if you trust the mounted plans.
- Use `EIG_FETCH_MOBILENET=0` (default) to skip Mobilenet assets when focusing on detection workloads.

## Latency + Debugging Tools

- `clients/python/examples/benchmark.py` measures request latency (mean/p50/p95/p99) and throughput.
- `clients/python/examples/detect_stream_yolov5.py` demonstrates streaming inference with a reusable TCP connection.
- Inspect server logs for per-request latency; they are structured JSON and easy to ingest with your logging stack.

For deeper profiling, run `trtexec --loadEngine=models/<engine>.plan --dumpProfile` inside the container to view TensorRT layer timings, or use Nsight Systems against the running gateway.

## Edge Orchestrator (Python)

The inference gateway now ships with an optional Python orchestrator that owns sensor ingestion, preprocessing, and agent execution around the C++ TensorRT server. Key points:
- Runs alongside the gateway (same container or host) and communicates via the existing TCP protocol.
- Provides connector plugins (MQTT, BLE, OpenCV camera) out of the box; new transports can be registered under `orchestrator/connectors`.
- Uses a connection pool to multiplex multiple pipelines across the TensorRT models configured in `config/models.yaml`.
- Dispatches actions (MQTT topics, webhooks, GPIO scripts) emitted by configurable agents.

Read `docs/SYSTEM_DESIGN.md` for pipeline configuration and deployment guidance for the orchestrator layer.

---
Maintainers: when touching the runtime, keep headers focused, avoid hidden magic in the entrypoint, and prefer explicit environment variable overrides for optional behavior. This keeps deployments predictable across embedded devices and laptops alike.
