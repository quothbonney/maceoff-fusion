"""G sweep for node fwd (graph replay x20 amortised, us) + correctness. usage: sweep.py OUT.json systems"""
import os, sys, json, time
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/nodetp8_fwd"); sys.path.insert(0, f"{R}/scripts/nodetp8_bwd")
import numpy as np, torch, ase.io
from mace.calculators import MACECalculator
from mace.tools.scatter import scatter_sum
import nodetp_fwd as NT
import csr as CSR
calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64")
model = calc.models[0]; torch.set_default_dtype(torch.float64)
cap = {}
for li, inter in enumerate(model.interactions):
    inter.register_forward_pre_hook(lambda m, a, kw, li=li: cap.setdefault(li, {}).update(
        ei=kw["edge_index"].detach().clone(), y=kw["edge_attrs"].detach().clone()), with_kwargs=True)
    inter.linear_up.register_forward_hook(lambda m, a, o, li=li: cap[li].update(x=o.detach().clone()))
    inter.conv_tp.register_forward_pre_hook(lambda m, a, li=li: cap[li].update(w=a[2].detach().clone()))
cfgs = {li: NT.make_cfg(model.interactions[li].conv_tp) for li in (0, 1)}
def gt(fn, k=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k): fn()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(50):
        t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / k * 1e6)
    return float(np.median(ts))
res = {}
for s in sys.argv[2].split(","):
    a = ase.io.read(f"{R}/results/workloads/systems/{s}.extxyz"); cap.clear(); a.calc = calc; a.get_potential_energy()
    for li in (0, 1):
        c = cap[li]; tp = model.interactions[li].conv_tp; N = c["x"].shape[0]
        with torch.no_grad():
            ref = scatter_sum(tp(c["x"][c["ei"][0]], c["y"], c["w"]), c["ei"][1], dim=0, dim_size=N)
        csr = CSR.build_csr(c["ei"], N)
        for PG, G in [(pg, g) for pg in (1, 2, 4) for g in (1, 2, 4) if pg * g <= 8 and not (li == 1 and pg == 1 and g > 2)]:
            o = NT.fwd(cfgs[li], c["x"], c["y"], c["w"], c["ei"], csr, G=G, PG=PG)
            e = float((o - ref).abs().max() / ref.abs().max())
            t = gt(lambda: NT.fwd(cfgs[li], c["x"], c["y"], c["w"], c["ei"], csr, out=o, G=G, PG=PG))
            res[f"{s}_L{li}_PG{PG}_G{G}"] = dict(us=t, rel=e, auto=NT.choose_G(N, c["ei"].shape[1]))
            print(s, li, "PG", PG, "G", G, f"{t:.1f}us rel={e:.2e} auto={NT.choose_G(N, c['ei'].shape[1])}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
