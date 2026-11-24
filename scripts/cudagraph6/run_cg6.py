"""cudagraph6 driver: capture feasibility, correctness and timing of GraphedMACE vs eager.
usage: python -I run_cg6.py OUT.json [systems,comma] [backends,comma]"""
import os, sys, json, time, math, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HB  # first: pins flashTP_e3nn to src/flashTP-graphcap
sys.path.insert(1, f"{R}/scripts/graph")
import numpy as np
import torch
import ase.io
from ase import Atoms
from mace.tools import torch_tools
from proto_graph import ProtoBuilder
from graphed_model import GraphedMACE, round_cap

assert HB.flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-graphcap/"), HB.flashTP_e3nn.__file__
out_path = sys.argv[1]
SYS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["aspirin", "org192", "water192", "org648", "water648", "wbox648",
                                                        "single_C", "far_pair"]
BACK = sys.argv[3].split(",") if len(sys.argv) > 3 else ["e3nn", "flash", "hybrid"]
WARM, REPS = 10, 60
dev = torch.device("cuda")


def load(name):
    if name == "single_C":
        return Atoms("C", positions=[[0, 0, 0]])
    if name == "far_pair":
        return Atoms("OH", positions=[[0, 0, 0], [12.0, 0, 0]])
    return ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")


def eager(model, bd):
    with torch_tools.default_dtype("float64"):
        o = model(bd, compute_force=True, training=False)
    return o["energy"].detach().clone(), o["forces"].detach().clone()


def cmp(E, F, Er, Fr, n):
    d = (F - Fr).abs()
    return dict(dE_per_atom=float((E - Er).abs().max()) / max(n, 1), max_dF=float(d.max()) if d.numel() else 0.0,
                rms_dF=float(((F - Fr) ** 2).sum(-1).mean().sqrt()) if d.numel() else 0.0,
                finite=bool(torch.isfinite(E).all() and torch.isfinite(F).all()))


def timeit(fn, warm=WARM, reps=REPS):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(n=reps, median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                min=float(a.min()))


def count_api(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    api, gpu = {}, 0
    for e in p.events():
        nm = e.name
        if nm.startswith("cuda") or nm.startswith("cu"):
            if any(s in nm for s in ("Launch", "Memcpy", "Memset")):
                api[nm] = api.get(nm, 0) + 1
        if str(e.device_type).endswith("CUDA"):
            gpu += 1
    return dict(api_calls=api, n_launch_api=sum(api.values()), n_gpu_ops=gpu)


def gpuinfo():
    import subprocess
    q = "index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()


res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), torch=torch.__version__,
           flashTP=HB.flashTP_e3nn.__file__, check_indices=False, warm=WARM, reps=REPS, systems={}, failures=[])
calcs = {}
calcs["e3nn"], _ = HB.build_calc(conv="e3nn", sc="e3nn", check_indices=False)
if "flash" in BACK:
    calcs["flash"], info = HB.build_calc(conv="flashtp", sc="e3nn", check_indices=False)
    res["flash_cg"] = info["flash_cg"]
if "hybrid" in BACK:
    calcs["hybrid"], info = HB.build_calc(conv="flashtp", sc="cueq", check_indices=False)
    res["hybrid_backend_map"] = info["backend_map"]
builders = {k: ProtoBuilder(c) for k, c in calcs.items()}
models = {k: c.models[0] for k, c in calcs.items()}
r_max = float(calcs["e3nn"].r_max)
res["r_max"] = r_max


def save():
    res["gpu_end"] = gpuinfo()
    json.dump(res, open(out_path, "w"), indent=1)


for sname in SYS:
    atoms0 = load(sname)
    n = len(atoms0)
    bd0 = builders["e3nn"].build(atoms0)
    E0 = int(bd0["edge_index"].shape[1])
    cap = round_cap(E0)
    S = res["systems"][sname] = dict(n_atoms=n, n_edges=E0, E_cap=cap, pbc=bool(any(atoms0.pbc)), backends={})
    Eref0, Fref0 = eager(models["e3nn"], bd0)
    # noisy sequence (cumulative 0.02 A gaussian), same for all backends
    rng = np.random.default_rng(0)
    seq = [atoms0]
    for k in range(5):
        a = seq[-1].copy(); a.positions = a.positions + rng.normal(0, 0.02, a.positions.shape); seq.append(a)
    refs = []
    for a in seq:
        b = builders["e3nn"].build(a)
        refs.append((b, *eager(models["e3nn"], b)))
    S["seq_edges"] = [int(b["edge_index"].shape[1]) for b, _, _ in refs]
    for bk in BACK:
        model, builder = models[bk], builders[bk]
        B = S["backends"][bk] = {}
        print(f"=== {sname} {bk} N={n} E={E0} cap={cap}", flush=True)
        try:
            # eager same backend, unpadded
            Ee, Fe = eager(model, bd0)
            B["eager_vs_e3nn_fixed"] = cmp(Ee, Fe, Eref0, Fref0, n)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
            G = GraphedMACE(model, bd0, cap, r_max)
            B["capture_s"] = G.capture_s
            B["peak_mem_capture_MiB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
            B["graph_resident_MiB"] = (torch.cuda.memory_allocated() - m0) / 2**20
            # correctness: fixed + sequence
            rows = []
            for i, (b, Er, Fr) in enumerate(refs):
                bb = builder.build(seq[i])
                Eg, Fg = G.step(bb)
                Eg, Fg = Eg.clone(), Fg.clone()
                Eb, Fb = eager(model, bb)
                rows.append(dict(step=i, E=int(bb["edge_index"].shape[1]), vs_e3nn=cmp(Eg, Fg, Er, Fr, n),
                                 vs_eager_same=cmp(Eg, Fg, Eb, Fb, n)))
            B["correctness_seq"] = rows
            # repeat variation (fixed geometry, 5 replays)
            G.step(bd0)
            E1, F1 = G.out_E.clone(), G.out_F.clone()
            rv = []
            for _ in range(4):
                G.replay(); rv.append(cmp(G.out_E, G.out_F, E1, F1, n))
            B["repeat_replay"] = dict(max_dE_per_atom=max(r["dE_per_atom"] for r in rv), max_dF=max(r["max_dF"] for r in rv))
            # padding parity: padded graphed e3nn result vs unpadded (fixed) already in rows[0]; also eager padded
            if bk == "e3nn":
                B["padded_eager_vs_unpadded"] = cmp(*eager(model, G.static), Eref0, Fref0, n)
            # timing
            fixed_atoms = seq[0]
            B["t_eager_model"] = timeit(lambda: eager(model, bd0))
            B["t_graphed_model_copyin_replay"] = timeit(lambda: G.step(bd0))
            B["t_graphed_replay_only"] = timeit(lambda: G.replay())
            B["t_e2e_eager_gpubuild"] = timeit(lambda: eager(model, builder.build(fixed_atoms)))
            B["t_e2e_graphed_gpubuild"] = timeit(lambda: G.step(builder.build(fixed_atoms)))
            B["t_gpubuild_only"] = timeit(lambda: builder.build(fixed_atoms))
            B["api_eager_model"] = count_api(lambda: eager(model, bd0))
            B["api_graphed_step"] = count_api(lambda: G.step(bd0))
            B["recaptures"] = G.recaptures
            # recapture cost: force growth beyond capacity
            t0 = time.perf_counter(); G2 = GraphedMACE(model, bd0, round_cap(E0, 0.25), r_max); B["recapture_s"] = time.perf_counter() - t0
            del G2
            del G
            torch.cuda.empty_cache()
            print(json.dumps({k: (v["median"] if isinstance(v, dict) and "median" in v else None) for k, v in B.items() if k.startswith("t_")}), flush=True)
        except Exception as ex:
            B["error"] = f"{type(ex).__name__}: {ex}"
            B["traceback"] = traceback.format_exc()
            res["failures"].append(f"{sname}/{bk}: {B['error'][:300]}")
            print(B["traceback"], flush=True)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        save()
save()
print("DONE failures:", res["failures"])
