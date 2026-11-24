"""graphed7 quick timing: GraphedRunner hybrid/flash. usage: python -I time_graphed.py OUT.json"""
import os, sys, json, time
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/graphed7")
import graphed_runner as GR
import numpy as np
import torch
import ase.io

out_path = sys.argv[1]
WARM, REPS = 15, 80
SYS = ["aspirin", "water192", "water648", "wbox648"]


def gpuinfo():
    import subprocess
    q = "index,name,uuid,clocks.sm,clocks.max.sm"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()


def timeit(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(n=REPS, median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))


res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), warm=WARM, reps=REPS,
           rows={})
for bk in ["hybrid", "flash"]:
    calc, model = GR.build_backend(bk)
    for s in SYS:
        a = ase.io.read(f"{R}/results/workloads/systems/{s}.extxyz")
        modes = [False, True] if s == "wbox648" else [False]
        for stress in modes:
            rn = GR.GraphedRunner(calc, model, stress=stress)
            rn.step(a)
            key = rn.key_of(a)
            bd = rn._build(a)
            row = dict(N=len(a), E=int(bd["edge_index"].shape[1]), E_cap=rn.graphs[key].E_cap, stress=stress)
            row["model_copyin_replay"] = timeit(lambda: rn._step_bd(key, a, bd))
            row["step_tensors_gpubuild"] = timeit(lambda: rn.step_tensors(a))
            row["step_full_numpy"] = timeit(lambda: rn.step(a))
            row["gpubuild_only"] = timeit(lambda: rn._build(a))
            row["stats"] = rn.stats()
            nm = f"{bk}/{s}" + ("/stress" if stress else "")
            res["rows"][nm] = row
            print(nm, {k: round(v["median"], 3) for k, v in row.items() if isinstance(v, dict) and "median" in v},
                  flush=True)
            del rn
            torch.cuda.empty_cache()
    del calc, model
    torch.cuda.empty_cache()
res["gpu_end"] = gpuinfo()
with open(out_path, "w") as f:
    json.dump(res, f, indent=1)
print("DONE")
