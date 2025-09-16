# SPDX-License-Identifier: Apache-2.0
import socket, struct, numpy as np, os, itertools

MAGIC=b"TRT\x01"; VERSION=1
_req_counter = itertools.count(1)

def _pack_hdr(req_id, model_id, tensors):
    H = struct.pack("<4sHHIII", MAGIC, VERSION, 0, len(model_id), len(tensors), 0)
    body = model_id.encode()
    for a in tensors:
        a = np.ascontiguousarray(a)
        if   a.dtype==np.float32: dt=0
        elif a.dtype==np.float16: dt=1
        elif a.dtype==np.int8:    dt=2
        elif a.dtype==np.int32:   dt=3
        else: raise ValueError("unsupported dtype")
        nd=a.ndim; dims=a.shape
        body += struct.pack("<BB", dt, nd)
        body += struct.pack("<%di"%nd, *dims)
        raw=a.tobytes()
        body += struct.pack("<I", len(raw))
        body += raw
    frame = H + body
    return struct.pack("<I", len(frame)) + frame

def infer(host, port, model_id, arrays, timeout=10.0):
    req_id = next(_req_counter)
    frame = _pack_hdr(req_id, model_id, arrays)
    with socket.create_connection((host,port), timeout=timeout) as s:
        s.sendall(frame)
        def recvn(n):
            buf=b""
            while len(buf)<n:
                chunk=s.recv(n-len(buf))
                if not chunk: raise OSError("short read")
                buf+=chunk
            return buf
        flen, = struct.unpack("<I", recvn(4))
        payload = recvn(flen)
        off=0
        rid, status, nout = struct.unpack_from("<III", payload, off); off+=12
        lens = [struct.unpack_from("<I", payload, off+i*4)[0] for i in range(nout)]
        off += 4*nout
        outs=[]; 
        for L in lens:
            outs.append(payload[off:off+L]); off+=L
        return status, outs
