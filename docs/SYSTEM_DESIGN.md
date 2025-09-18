# Edge Inference Gateway System Design

This document expands on `docs/ARCHITECTURE.md` and describes the end-to-end platform we ship as the "edge infer gateway": how sensor data is collected, how models are orchestrated, and how agents close the decision loop in sub-second latency budgets.

## Goals
- Serve multiple TensorRT models concurrently on resource-constrained NVIDIA GPUs (e.g. Jetson, GeForce 920MX laptops).
- Ingest heterogeneous sensor streams (BLE, Wi-Fi/MQTT, Ethernet/RTSP) through a unified, low-latency pipeline.
- Allow end users to declaratively describe pipelines and agents that map sensor events to model invocations and actions.
- Provide observability (health, metrics, tracing hooks) and operational guardrails (backpressure, retries, fallbacks) required for industrial IoT deployments.

## High-Level Topology

```
┌──────────────┐    radio/wired     ┌──────────────────┐        TCP        ┌────────────────────┐
│ Sensor Nodes │ ─────────────────▶ │ Edge Orchestrator│ ───────────────▶ │ TensorRT Inference │
│ (BLE, MQTT,  │                    │  (pipelines +     │                 │    Gateway (C++)    │
│  cameras)    │ ◀──────── actions ─│  agents, actions) │ ◀─────────────── │  (this repo core)   │
└──────────────┘                    └──────────────────┘                  └────────────────────┘
       ▲                                                                          │
       │                                                                          ▼
       │                                                                   Prom/HTTP/Logs
       │
┌──────────────┐
│ Actuators &  │
│ Enterprise   │
│ Systems      │
└──────────────┘
```

The orchestrator is new: it runs alongside the existing TCP gateway and is responsible for connecting wireless/wired sensors, executing preprocessing, selecting models, aggregating outputs, and triggering configurable agents.

## Component Overview

### Sensor Nodes
- Arduino Nano 33 BLE Sense: publishes IMU/environment metrics over BLE or Wi-Fi gateway.
- ESP32 (Wi-Fi) modules: push telemetry or camera JPEGs over MQTT or HTTP.
- Raspberry Pi with camera: streams frames via RTSP/WebRTC or pushes JPEG bursts via MQTT.
- Wired PLCs or Modbus sensors can use the REST/serial connector.

### Edge Orchestrator (Python)
- Async runtime (built on `asyncio`) that hosts:
  - **Connector plugins**: MQTT, USB/CSI cameras (OpenCV), BLE (Bleak). The registry can be extended by adding modules under `orchestrator/connectors`. Each normalises incoming payloads into an internal `EdgeMessage` structure.
  - **Pipeline engine**: resolves the configured pipeline for each message, runs preprocessing and inference, and tracks deadlines.
  - **Gateway pool**: maintains a pool of TCP connections to the C++ TensorRT server to keep inference latency deterministic.
  - **Agent runtime**: executes user-defined decision logic (Python classes) that consume postprocessed results and emit actions (e.g. MQTT command, HTTP webhook, GPIO toggle).
  - **Metrics HTTP server**: exports orchestrator health, queue depth, inference latency, and agent success counts.

### TensorRT Inference Gateway (C++)
- Existing epoll-based TCP server. No architectural changes required—multi-model support, metrics, and concurrency are reused.
- Only change is the introduction of configuration describing which models the orchestrator can call.

### Action Dispatchers
- MQTT publisher for ESP32/Arduino actuators.
- REST webhook dispatcher (for SCADA, MES systems, Teams/Slack alerts).
- Local command runner (e.g. `gpiozero` script) for direct actuation.

## Data Flow Lifecycle

1. **Acquire**: A connector receives sensor data (`EdgeMessage`). Timestamp and origin metadata are attached upstream to avoid clock drift issues.
2. **Preprocess**: The pipeline applies a named preprocessing function (e.g. image resize/normalise, sensor vector validation). Preprocessors run in the orchestrator process and can be GPU-aware if CUDA is available.
3. **Inference**: The orchestrator borrows a socket from the gateway pool and performs inference by marshalling tensors through the binary TCP protocol.
4. **Postprocess**: Raw outputs are translated into semantic objects (bounding boxes, classification top-k, anomaly scores).
5. **Agent decisions**: Configured agents receive structured context and may emit actions. Agents are fully async to avoid blocking other pipelines.
6. **Dispatch**: Actions are fan-out to the appropriate dispatcher (MQTT, REST, file log, etc.).
7. **Observe**: Every stage records structured events via the orchestrator metrics/log pipeline for troubleshooting and SLA verification.

## Configuration Model (`config/pipelines.yaml`)

```yaml
version: 1

gateway:
  host: 127.0.0.1
  port: 8008
  pool_size: 4         # concurrent sockets to the TensorRT gateway
  timeout_s: 3.0

connectors:
  - id: floor1-mqtt
    type: mqtt
    host: 192.168.10.40
    port: 1883
    username: edge
    password: changeme
    topics:
      - filter: sensors/floor1/+/env
        pipeline: env-quality
        serializer: json
      - filter: sensors/floor1/cam1/frame
        pipeline: frontdoor-vision
        serializer: jpeg

pipelines:
  - id: env-quality
    preprocess: env.vector_to_tensor
    model: mobilenet_v2_cls
    postprocess: env.softmax_topk
    agents:
      - air_quality_alert

  - id: frontdoor-vision
    preprocess: vision.jpeg_to_yolov5
    model: yolov5n_coco
    postprocess: vision.yolo_nms
    agents:
      - frontdoor_guard
      - frontdoor_archive

agents:
  air_quality_alert:
    type: threshold
    metric: co2_ppm
    threshold: 800
    dispatcher: log
  frontdoor_guard:
    type: person_in_zone
    zone: frontdoor
    dispatcher: mqtt_actuators
    target: actuators/frontdoor
  frontdoor_archive:
    type: snapshot_archive
    dispatcher: log

actions:
  log:
    type: log
  mqtt_actuators:
    type: mqtt
    host: floor1-mqtt
    topic: actuators/frontdoor
```

- **connectors**: Each connector binds to its transport and yields messages. Multiple topics may map to different pipelines.
- **pipelines**: Reference preprocessing/postprocessing callables in `orchestrator/plugins` and list agent names to execute per event.
- **agents**: Defined once under `agents:` with a `type` and options; pipelines reference them by key, enabling reuse across multiple sensor routes.
- **actions**: Dispatcher definitions (`log`, `mqtt`, `webhook`, ...). Agents refer to these by dispatcher name via `Action.dispatcher` when emitting commands.

## Latency & Determinism Strategies
- **Zero-copy tensors**: Preprocessors allocate contiguous NumPy arrays in the correct dtype/layout to avoid conversions in the TensorRT gateway.
- **Connection pooling**: Reuse live TCP sockets and TensorRT contexts to avoid cold start penalties.
- **Deadline-aware pipelines**: Each pipeline may declare a `deadline_ms`. The orchestrator will drop or downgrade requests that exceed the budget to keep the system responsive.
- **Backpressure**: If a connector overwhelms a pipeline, the orchestrator can shed load (configurable `max_queue_depth`) or publish a throttle command back to the sensor node.

## Reliability Considerations
- Connectors reconnect with exponential backoff.
- Inference failures trigger retry on a fresh socket; repeated failures trip a circuit breaker and notify observability.
- Agent exceptions are captured and surfaced via metrics without killing the pipeline loop.
- Orchestrator and TensorRT gateway expose `/healthz` and `/metrics`; deployments can attach liveness probes for Kubernetes or systemd.

## Extensibility
- Add new connectors by implementing `BaseConnector` (async iterator returning `EdgeMessage`).
- Add preprocessors/postprocessors/agents by dropping Python modules in `orchestrator/plugins/` and referencing by dotted path in YAML.
- Multi-GPU laptops can run several instances of the TensorRT gateway (one per GPU) with the orchestrator sharding pipelines via `model_affinity` rules in config.
- Edge nodes that already run local ML can be integrated by turning them into upstream connectors and sharing inference results, enabling federated agent logic.

## Validation Paths
- `tools/simulate_sensor.py` can publish synthetic trajectories to MQTT to validate pipelines without hardware.
- `clients/python/examples/benchmark.py` remains the reference for measuring model latency once the orchestrator is online.
- Integration tests spin up the orchestrator, a loopback MQTT broker (using `gmqtt`/`asyncio-mqtt` test server), and the TensorRT gateway in-process to assert end-to-end latency contracts.

```
