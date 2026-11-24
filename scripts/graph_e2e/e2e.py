"""End-to-end: stock MACECalculator energy+forces call vs the same backend fed by the GPU graph prototype
(scripts/graph/proto_graph.py). Backends: e3nn, flash (FlashTP-FP64CG v2), hybrid (v2). fp64. Interleaved.
Timed boundaries (cuda-synchronized):
  stock_ase : calc.reset(); atoms.get_potential_energy(); atoms.get_forces()
  proto_e2e : bd = PB.build(atoms); out = model(bd, compute_force=True, training=False);
              E = float(out['energy']); F = out['forces'].detach().cpu().numpy()   (same work the calculator does after its graph build)
Usage: python -I e2e.py <out_json> [systems]"""
import os, sys, json, time, subprocess
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_v2"); import hybrid_v2 as HB   # must be first (flashTP_e3nn from cgpatch)
sys.path.insert(0, f"{R}/scripts/graph"); sys.path.insert(0, f"{R}/scripts/flash_proto")
import numpy as np, torch
import proto_graph
from mace.tools import torch_tools
from systems import get_system, water_cluster
M = f"{R}/models/MACE-OFF23_medium.model"
WARM, BLOCK, NBLK = 10, 10, int(os.environ.get("NBLK", 5))
sync, pc = torch.cuda.synchronize, time.perf_counter

def water_box(n_side=6, density=1.0, seed=1):   # identical to scripts/graph/graph_profile.py
    nmol = n_side ** 3; vol = nmol * 18.01528 * 1.66053906660e-24 / density * 1e24
    L = vol ** (1 / 3); sp = L / n_side
    a = water_cluster(n_side, spacing=sp, seed=seed); a.positions += sp / 2; a.set_cell([L, L, L]); a.set_pbc(True); return a

def system(n): return water_box(6) if n == "waterbox648_pbc" else get_system(n)
def nvsmi(): return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
def st(x):
    x = np.asarray(x) * 1e3; p10, p50, p90 = np.percentile(x, [10, 50, 90]); return dict(n=int(x.size), median=float(p50), p10=float(p10), p90=float(p90), min=float(x.min()))

names = sys.argv[2:] or ["H2O", "water192", "water648", "water1536", "waterbox648_pbc"]
B = {"e3nn": HB.build_calc(M, conv="e3nn", sc="e3nn", check_indices=False)[0],
     "flash": HB.build_calc(M, conv="flashtp", sc="e3nn", check_indices=False)[0],
     "hybrid": HB.build_calc(M, conv="flashtp", sc="cueq", check_indices=False)[0]}
PB = {k: proto_graph.ProtoBuilder(c) for k, c in B.items()}
out = dict(job=os.environ.get("SLURM_JOB_ID"), nvsmi_start=nvsmi(), check_indices=False, warm=WARM, block=BLOCK, nblk=NBLK, systems={})

def stock(calc, atoms):
    atoms.calc = calc; calc.reset(); e = atoms.get_potential_energy(); f = atoms.get_forces(); return e, f
def proto(k, atoms):
    bd = PB[k].build(atoms)
    with torch_tools.default_dtype("float64"):
        o = B[k].models[0](bd, compute_force=True, training=False)
    return float(o["energy"].detach().sum()), o["forces"].detach().cpu().numpy()

for name in names:
    atoms = system(name); res = {"n_atoms": len(atoms), "pbc": bool(atoms.pbc.any())}
    ref_e, ref_f = stock(B["e3nn"], atoms.copy())
    rng = np.random.default_rng(7); seq = [atoms.positions + rng.normal(scale=0.02, size=atoms.positions.shape) for _ in range(3)]
    par = {}
    for k in B:   # parity: proto path vs stock e3nn, fixed + changing coordinates
        dE, dF, rms = [], [], []
        for p in [atoms.positions] + seq:
            a = atoms.copy(); a.positions = p
            e0, f0 = stock(B["e3nn"], a.copy()); e1, f1 = proto(k, a)
            dE.append(abs(e1 - e0) / len(a)); dF.append(np.abs(f1 - f0).max()); rms.append(np.sqrt(((f1 - f0) ** 2).mean()))
        par[k] = dict(max_dE_per_atom=float(max(dE)), max_dF=float(max(dF)), max_rmsdF=float(max(rms)))
    res["parity_proto_vs_stock_e3nn"] = par
    ok = all(v["max_dE_per_atom"] < 1e-11 and v["max_dF"] < 1e-12 for v in par.values())
    res["parity_ok"] = ok
    if not ok:
        print(name, "PARITY FAIL", par, flush=True); out["systems"][name] = res; continue
    T = {f"{k}_{m}": [] for k in B for m in ("stock", "proto")}
    fns = {}
    for k, c in B.items():
        a_s = atoms.copy(); a_p = atoms.copy()
        fns[f"{k}_stock"] = (lambda c=c, a=a_s: stock(c, a)); fns[f"{k}_proto"] = (lambda k=k, a=a_p: proto(k, a))
    for f in fns.values():
        for _ in range(WARM): f()
    order = list(fns)
    for b in range(NBLK):
        for key in (order if b % 2 == 0 else order[::-1]):
            for _ in range(BLOCK):
                sync(); t = pc(); fns[key](); sync(); T[key].append(pc() - t)
    res["timing_ms"] = {k: st(v) for k, v in T.items()}
    res["speedup_vs_e3nn_stock"] = {k: res["timing_ms"]["e3nn_stock"]["median"] / v["median"] for k, v in res["timing_ms"].items()}
    out["systems"][name] = res
    print(name, len(atoms), json.dumps({k: round(v["median"], 2) for k, v in res["timing_ms"].items()}), "parity", json.dumps(par), flush=True)
out["nvsmi_end"] = nvsmi()
json.dump(out, open(sys.argv[1], "w"), indent=2)
