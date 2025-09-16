# SPDX-License-Identifier: Apache-2.0
import socket, struct
import numpy as np

MAGIC=b"TRT\x01"; VERSION=1

def infer(host, port, model_id: str, arrs):
    # arrs: list[np.ndarray] (contiguous, C-order)
    hdr = struct.pack("<4sHHIII", MAGIC, VERSION, 0, len(model_id), len(arrs), 0)
    body = model_id.encode()
    # each input: dtype(1) ndims(1) dims(int32*nd) len(uint32)  + raw
    payload = []
    for a in arrs:
        a = np.ascontiguousarray(a)
        if   a.dtype==np.float32: dt=0
        elif a.dtype==np.float16: dt=1
        elif a.dtype==np.int8:    dt=2
        elif a.dtype==np.int32:   dt=3
        else: raise ValueError("unsupported dtype")
        nd = a.ndim
        dims = list(a.shape)
        raw = a.tobytes()
        desc = struct.pack("<BB", dt, nd) + struct.pack("<%di"%nd, *dims) + struct.pack("<I", len(raw))
        body += desc
        payload.append(raw)

    with socket.create_connection((host,port), timeout=10) as s:
        s.sendall(hdr + body + b"".join(payload))
        # response: status(uint32) nout(uint32) each: len(uint32) then raw blobs
        def recvn(n):
            buf=b""
            while len(buf)<n:
                chunk=s.recv(n-len(buf))
                if not chunk: raise OSError("short read")
                buf+=chunk
            return buf
        status, nout = struct.unpack("<II", recvn(8))
        lens = [struct.unpack("<I", recvn(4))[0] for _ in range(nout)]
        outs  = [recvn(L) for L in lens]
    return outs
