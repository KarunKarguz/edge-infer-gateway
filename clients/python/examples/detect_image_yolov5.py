import argparse, numpy as np, cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from gateway_client import infer

def letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    shape = img.shape[:2]  # hw
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2; dh //= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def nms(boxes, scores, iou_th=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        ious = iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return keep

def iou_xyxy(b0, B):
    # b0: (4,), B: (N,4)
    inter_x1 = np.maximum(b0[0], B[:,0])
    inter_y1 = np.maximum(b0[1], B[:,1])
    inter_x2 = np.minimum(b0[2], B[:,2])
    inter_y2 = np.minimum(b0[3], B[:,3])
    inter_w = np.maximum(0, inter_x2-inter_x1)
    inter_h = np.maximum(0, inter_y2-inter_y1)
    inter = inter_w*inter_h
    area0 = (b0[2]-b0[0])*(b0[3]-b0[1])
    areaB = (B[:,2]-B[:,0])*(B[:,3]-B[:,1])
    return inter / (area0 + areaB - inter + 1e-6)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model", default="yolov5n_coco")
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", default="assets/coco.names")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    names = [l.strip() for l in Path(args.labels).read_text().splitlines()]

    # BGR image
    img0 = cv2.imread(args.image)
    assert img0 is not None, f"failed to read image: {args.image}"
    ih, iw = img0.shape[:2]
    img, r, (dw, dh) = letterbox(img0, (640,640))
    x = img[:, :, ::-1].transpose(2,0,1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    x = x[None, ...]

    # YOLOv5 engines here are built as FP16; send FP16 for best perf
    status, outs = infer(args.host, args.port, args.model, [x.astype(np.float16)])
    assert status == 0, f"infer status={status}"
    pred = np.frombuffer(outs[0], dtype=np.float16).astype(np.float32).reshape(1,-1,85)[0]

    # boxes xywh -> xyxy in padded image space
    boxes = pred[:, :4]
    scores_obj = 1/(1+np.exp(-pred[:, 4]))
    cls_logits = pred[:, 5:]
    cls_scores = 1/(1+np.exp(-cls_logits))
    cls_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores.max(axis=1)
    conf = scores_obj * cls_conf
    mask = conf >= args.conf
    boxes = boxes[mask]; conf = conf[mask]; cls_ids = cls_ids[mask]
    if boxes.size == 0:
        print("No detections above threshold.")
        exit(0)
    # xywh -> xyxy
    xyxy = np.zeros_like(boxes)
    xyxy[:,0] = boxes[:,0] - boxes[:,2]/2
    xyxy[:,1] = boxes[:,1] - boxes[:,3]/2
    xyxy[:,2] = boxes[:,0] + boxes[:,2]/2
    xyxy[:,3] = boxes[:,1] + boxes[:,3]/2

    # scale back to original image coords
    gain = min(640/ih, 640/iw)
    pad_x, pad_y = (640 - iw*gain)/2, (640 - ih*gain)/2
    xyxy[:, [0,2]] -= pad_x
    xyxy[:, [1,3]] -= pad_y
    xyxy /= gain
    xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, iw)
    xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, ih)

    # NMS
    keep = nms(xyxy, conf, args.iou)
    xyxy = xyxy[keep]; conf = conf[keep]; cls_ids = cls_ids[keep]

    for (x1,y1,x2,y2), c, k in zip(xyxy.astype(int), conf, cls_ids):
        cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{names[int(k)]}:{c:.2f}"
        cv2.putText(img0, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    out_path = str(Path(args.image).with_suffix(".yolo_out.jpg"))
    cv2.imwrite(out_path, img0)
    print(f"Wrote detections to {out_path}")
