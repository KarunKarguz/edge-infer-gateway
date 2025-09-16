import argparse, time, numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from gateway_stream import GatewayStream

def rand_input(model: str):
    if model.startswith("yolov5"):
        x = np.random.rand(1,3,640,640).astype(np.float16)
    elif model.startswith("ssd"):
        x = np.random.rand(1,3,300,300).astype(np.float32)
    else:
        x = np.random.rand(1,3,224,224).astype(np.float32)
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model", default="yolov5n_coco")
    ap.add_argument("--warmup", "--warmup-iters", dest="warmup", type=int, default=10,
                    help="warmup iterations before timing")
    ap.add_argument("--count", "--iters", "--iter", dest="count", type=int, default=100,
                    help="number of timed iterations")
    args = ap.parse_args()

    gs = GatewayStream(args.host, args.port)
    x = rand_input(args.model)
    # warmup
    for _ in range(args.warmup):
        status, _ = gs.infer(args.model, [x])
        assert status == 0
    t0 = time.time()
    timings = []
    for _ in range(args.count):
        t = time.perf_counter()
        status, _ = gs.infer(args.model, [x])
        assert status == 0
        timings.append((time.perf_counter() - t) * 1000.0)
    dt = time.time() - t0
    arr = np.array(timings)
    mean = arr.mean()
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    print(f"count={args.count} total={dt:.3f}s qps={args.count/dt:.2f}")
    print(f"latency_ms: mean={mean:.2f} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}")
    gs.close()
