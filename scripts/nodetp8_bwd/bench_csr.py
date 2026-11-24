import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csr import build_csr

def v_int32(ei, N):
    out = {}
    nodes = torch.arange(N + 1, device=ei.device, dtype=torch.int32)
    for name, row in (("dst", ei[1]), ("src", ei[0])):
        k, p = torch.sort(row.to(torch.int32), stable=True)
        out[f"perm_{name}"] = p
        out[f"rowptr_{name}"] = torch.searchsorted(k, nodes, out_int32=True)
    return out

def v_cnt(ei, N):
    # counting-sort style: rowptr from searchsorted of sorted keys, but argsort via one combined int64 key sort
    out = {}
    E = ei.shape[1]
    nodes = torch.arange(N + 1, device=ei.device, dtype=torch.int64)
    for name, row in (("dst", ei[1]), ("src", ei[0])):
        key = row * E + torch.arange(E, device=ei.device)  # unique keys -> unstable sort is fine
        k = torch.sort(key, stable=False).values
        out[f"perm_{name}"] = k % E
        out[f"rowptr_{name}"] = torch.searchsorted(k, nodes * E, out_int32=True)
    return out

def gtime(fn, reps=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    ts = []
    for _ in range(20):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / reps)
    return round(float(np.median(ts)), 1)

for N, E in [(21, 302), (192, 6634), (648, 35698)]:
    src = torch.sort(torch.randint(0, N, (E,), device="cuda")).values
    ei = torch.stack([src, torch.randint(0, N, (E,), device="cuda")])
    ref = build_csr(ei, N)
    for nm, f in [("cur", build_csr), ("int32", v_int32), ("ukey", v_cnt)]:
        o = f(ei, N)
        ok = all(torch.equal(o[k].long(), ref[k].long()) for k in ref)
        print(N, E, nm, gtime(lambda: f(ei, N)), ok, flush=True)
