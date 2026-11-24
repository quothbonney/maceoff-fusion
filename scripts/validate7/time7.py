"""validate7 performance gate. usage: python -I time7.py OUT.json configs(comma) systems(comma)
configs: <backend>[+embed][+node]. All configs are built in one process (build+install+compile time recorded), then for each
system the configs are timed back to back (order rotated per system). Per (system, config):
  first_step_s   : first runner.step_tensors incl. graph build + capture (+ any lazy JIT on first capture)
  t_replay       : graph.replay() only, synced           (>=10 warmups, 60 reps; median, p10, p90)
  t_step         : runner.step_tensors(atoms) synced = GPU graph build (proto builder) + copy-in + replay
  n_kernels      : GPU ops inside the graph (profiler on 1 replay); fused kernel time + fraction of replay wall
  peak_MiB       : torch max_memory_allocated delta across capture+timing for this config/system
"""
import os, sys, json, time, traceback, collections
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/graphed7")
import graphed_runner as GR  # first
import numpy as np
import torch
import ase.io
from ase.build import molecule
from torch.profiler import profile, ProfilerActivity

out_path = sys.argv[1]; CFG = sys.argv[2].split(","); SYS = sys.argv[3].split(",")
WARM, REPS = 10, 60
FUSED_PAT = ("fe_sh", "fe_radial", "seg_linear")


def load(n):
    return molecule("H2O") if n == "H2O" else ase.io.read(f"{R}/results/workloads/systems/{n}.extxyz")


def timeit(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), n=REPS)


def gpuinfo():
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), warm=WARM, reps=REPS,
           configs={}, systems={})
built = {}
for c in CFG:
    parts = c.split("+"); tr = []
    t0 = time.perf_counter()
    if "embed" in parts:
        sys.path.insert(0, f"{R}/scripts/fuse7_embed"); import fused_embed as FE; tr.append(FE.install)
    if "node" in parts:
        sys.path.insert(0, f"{R}/scripts/fuse7_node"); import fused_node as FN; tr.append(FN.install)
    calc, model = GR.build_backend(parts[0], transforms=tr)
    torch.cuda.synchronize()
    res["configs"][c] = dict(build_install_s=time.perf_counter() - t0)
    built[c] = (calc, model)
    print("built", c, res["configs"][c], flush=True)

for si, s in enumerate(SYS):
    atoms = load(s)
    order = CFG[si % len(CFG):] + CFG[:si % len(CFG)]
    S = res["systems"][s] = dict(N=len(atoms), order=order)
    for c in order:
        calc, model = built[c]
        try:
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
            rn = GR.GraphedRunner(calc, model)
            t0 = time.perf_counter(); o = rn.step_tensors(atoms); torch.cuda.synchronize(); first = time.perf_counter() - t0
            key = rn.key_of(atoms); g = rn.graphs[key]
            D = dict(first_step_s=first, capture_s=rn.stats()["last_capture_s"], E=o["E"], E_cap=o["E_cap"])
            D["t_replay"] = timeit(g.replay)
            D["t_step"] = timeit(lambda: rn.step_tensors(atoms))
            for _ in range(3): g.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
            ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            D["n_gpu_ops"] = len(ks)
            D["kernel_sum_us"] = sum(e.device_time for e in ks) if ks and hasattr(ks[0], "device_time") else sum(e.cuda_time for e in ks)
            fz = collections.Counter()
            for e in ks:
                if any(pp in e.name for pp in FUSED_PAT):
                    fz[e.name[:60]] += getattr(e, "device_time", 0) or e.cuda_time
            D["fused_kernels_us"] = dict(fz); D["fused_us"] = sum(fz.values())
            D["fused_frac_of_replay"] = D["fused_us"] / 1e3 / D["t_replay"]["median"]
            D["peak_MiB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
            S[c] = D
            del rn, g
            print(s, c, json.dumps(dict(replay=round(D["t_replay"]["median"], 4), step=round(D["t_step"]["median"], 4),
                                        nk=D["n_gpu_ops"], first=round(first, 3), fused_us=round(D["fused_us"], 1),
                                        peak=round(D["peak_MiB"]))), flush=True)
        except Exception as ex:
            S[c] = dict(error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc()); print(S[c]["tb"], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
res["gpu_end"] = gpuinfo()
json.dump(res, open(out_path, "w"), indent=1)
print("DONE", flush=True)
