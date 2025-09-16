import argparse, numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from gateway_client import infer

IMNET_MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
IMNET_STD  = np.array([0.229,0.224,0.225],dtype=np.float32)

def load_image_224(p):
    img = Image.open(p).convert("RGB").resize((224,224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)/255.0
    x = (x - IMNET_MEAN)/IMNET_STD
    x = np.transpose(x, (2,0,1))         # HWC -> CHW
    x = np.expand_dims(x, 0)             # NCHW
    return x

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model", default="mobilenet_v2_cls")
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", required=True)
    args=ap.parse_args()

    x = load_image_224(args.image)
    outs = infer(args.host,args.port,args.model,[x.astype(np.float32)])
    logits = np.frombuffer(outs[0], dtype=np.float32).reshape(1,1000)
    probs = np.exp(logits - logits.max()) ; probs /= probs.sum()
    top5 = probs[0].argsort()[-5:][::-1]
    labels = [l.strip() for l in Path(args.labels).read_text().splitlines()]
    print("\nTop-5:")
    for k in top5:
        print(f"{k:4d}: {probs[0,k]:7.4f}  {labels[k]}")
