import argparse, numpy as np, cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from gateway_stream import GatewayStream

def preprocess_ssd(img):
    # Resize 300x300, BGR->RGB, normalize to [0,1]
    img = cv2.resize(img, (300,300), interpolation=cv2.INTER_LINEAR)
    x = img[:, :, ::-1].transpose(2,0,1)  # to CHW RGB
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None, ...]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model", default="ssd_mobilenet_coco")
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # Open source
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"failed to open source: {src}"

    gs = GatewayStream(args.host, args.port)
    while True:
        ok, frame = cap.read()
        if not ok: break
        ih, iw = frame.shape[:2]
        x = preprocess_ssd(frame)
        status, outs = gs.infer(args.model, [x.astype(np.float32)])
        if status != 0:
            print("infer error", status); break
        det = np.frombuffer(outs[0], dtype=np.float32).reshape(1,1,200,7)[0,0]
        for d in det:
            _, cls, conf, x1, y1, x2, y2 = d
            if conf < args.conf: continue
            x1 = int(x1 * iw); y1=int(y1*ih); x2=int(x2*iw); y2=int(y2*ih)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame, f"{int(cls)}:{conf:.2f}", (x1,max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        if args.show:
            cv2.imshow("SSD Stream", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release(); gs.close()
    if args.show:
        cv2.destroyAllWindows()
