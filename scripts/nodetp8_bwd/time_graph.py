"""CUDA-graph-replay timing (x20 replays per graph, amortised) on identical real inputs:
node bwd kernel, build_csr, node fwd(B)+bwd(C) via autograd, FlashTP-graphcap fwd+bwd via autograd."""
import os, sys, json
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from ase.io import read
import nodeconv as NC, nodetp_bwd
from csr import build_csr

calc, info = HB.build_calc(conv="e3nn", sc="e3nn", check_indices=False)
model = calc.models[0]
dev = torch.device("cuda")
h = NC.install(model, fwd="auto")
convs = NC.node_convs(model)
fwd_fn = convs[0].fwd_fn
cap = {}
for nm in ["aspirin", "org192", "water192", "org648", "water648", "wbox648"]:
    for c in convs:
        c.record = []
    a = read(f"{R}/results/workloads/systems/{nm}.extxyz"); a.calc = calc; a.get_potential_energy()
    cap[nm] = [c.record[0] for c in convs]
for c in convs:
    c.record = None
flash = [HB.FA.FlashConv(model.interactions[li].conv_tp.orig, torch.float64, dev, check_indices=False) for li in range(2)]


def gtime(fn, reps=20, trials=30):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(trials):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / reps)
    ts = np.array(ts)
    return [round(float(np.median(ts)), 1), round(float(np.percentile(ts, 10)), 1), round(float(np.percentile(ts, 90)), 1)]


res = {}
for nm, recs in cap.items():
    for li, (x, y, w, ei) in enumerate(recs):
        cfg = convs[li].cfg
        N = x.shape[0]
        g = torch.randn((N, cfg.d_out), dtype=torch.float64, device=dev)
        csr = build_csr(ei, N)
        ct = (csr["perm_dst"], csr["rowptr_dst"], csr["perm_src"], csr["rowptr_src"])
        xr, yr, wr = (t.clone().requires_grad_() for t in (x, y, w))
        r = {}
        r["build_csr"] = gtime(lambda: build_csr(ei, N))
        r["node_bwd"] = gtime(lambda: nodetp_bwd.bwd(cfg, g, x, y, w, ei, csr))
        r["node_fwd_B"] = gtime(lambda: fwd_fn(cfg, x, y, w, ei, csr))

        def node_fb():
            o = NC.NodeConvFunction.apply(xr, yr, wr, ei, ct, cfg, fwd_fn)
            return torch.autograd.grad(o, (xr, yr, wr), g)

        def flash_fb():
            o = flash[li](xr, yr, wr, ei)
            return torch.autograd.grad(o, (xr, yr, wr), g)
        r["node_fwd+bwd"] = gtime(node_fb)
        r["flash_fwd"] = gtime(lambda: flash[li](x, y, w, ei))
        r["flash_fwd+bwd"] = gtime(flash_fb)
        r["flash_bwd_est(fb-f)"] = round(r["flash_fwd+bwd"][0] - r["flash_fwd"][0], 1)
        res[f"{nm}/L{li}"] = dict(N=N, E=int(ei.shape[1]), **r)
        print(nm, li, N, ei.shape[1], r, flush=True)
json.dump(dict(unit="us per call, CUDA-graph replay of 20 calls, median [p10,p90] over 30 replays", fwd=convs[0].fwd_name,
               results=res), open(f"{R}/results/nodetp8_bwd/time_graph.json", "w"), indent=1)
print("DONE")
