"""prof8 FlashTP driver. usage: python -I flash_drv.py capture|run|time [systems]
capture: run MACE-OFF23 medium fp64 (FlashTP-graphcap conv, e3nn SC) on each system, hook FlashConv inputs -> results/prof8/inputs_<sys>.pt
run    : for each system and layer: NITER x (fwd + bwd) of the FlashTP kernel on captured inputs (for ncu)
time   : CUDA-event timings of fwd/bwd per layer + torch.profiler kernel names/times + launch config
"""
import os, sys, json, time
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HG  # FIRST
import torch, ase.io
from ase.build import molecule

mode = sys.argv[1]
SYS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["aspirin", "water192", "org192", "water648", "org648"]
OUT = f"{R}/results/prof8"
calc, info = HG.build_calc(conv="flashtp", sc="e3nn", check_indices=False)
model = calc.models[0]
convs = [b.conv_tp for b in model.interactions]

def load(n):
    return molecule("H2O") if n == "H2O" else ase.io.read(f"{R}/results/workloads/systems/{n}.extxyz")

if mode == "capture":
    for s in SYS:
        cap = {}
        hs = [c.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, [t.detach().clone() for t in a]))
              for i, c in enumerate(convs)]
        at = load(s); at.calc = calc; at.get_potential_energy()
        for h in hs: h.remove()
        torch.save({i: cap[i] for i in cap}, f"{OUT}/inputs_{s}.pt")
        print(s, {i: [tuple(t.shape) for t in cap[i]] for i in cap}, flush=True)
    sys.exit()

def prep(s, i):
    x, y, w, ei = [t.cuda() for t in torch.load(f"{OUT}/inputs_{s}.pt")[i]]
    x, y, w = [t.requires_grad_(True) for t in (x, y, w)]
    g = torch.randn(x.shape[0], convs[i].out_dim, dtype=torch.float64, device="cuda")
    return x, y, w, ei, g

def step(i, x, y, w, ei, g):
    out = convs[i](x, y, w, ei)
    gx, gy, gw = torch.autograd.grad(out, (x, y, w), g)
    return out

if mode == "run":
    NITER = int(os.environ.get("NITER", "2"))
    for s in SYS:
        for i in range(2):
            a = prep(s, i)
            for _ in range(NITER): step(i, *a)
            torch.cuda.synchronize()
            print("ran", s, i, flush=True)
    sys.exit()

if mode == "time":
    from torch.profiler import profile, ProfilerActivity
    import numpy as np
    res = dict(job=os.environ.get("SLURM_JOB_ID"), flash_cg=info["flash_cg"], systems={})
    for s in SYS:
        for i in range(2):
            x, y, w, ei, g = prep(s, i)
            out = convs[i](x, y, w, ei)
            for _ in range(5): step(i, x, y, w, ei, g)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(20): step(i, x, y, w, ei, g)
                torch.cuda.synchronize()
            agg = {}
            for e in p.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    agg.setdefault(e.name, []).append(getattr(e, "device_time", 0) or e.cuda_time)
            ks = {k: dict(n=len(v) / 20, median_us=float(np.median(v))) for k, v in agg.items()}
            res["systems"][f"{s}/L{i}"] = dict(N=x.shape[0], E=ei.shape[1], kernels=ks)
            print(s, i, x.shape[0], ei.shape[1], json.dumps({k[:50]: round(v["median_us"], 1) for k, v in ks.items() if "sptp" in k}), flush=True)
    json.dump(res, open(f"{OUT}/flash_kernel_times.json", "w"), indent=1)
