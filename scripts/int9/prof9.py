"""int9 B2 profile.  usage: python -I prof9.py OUT.json modes(comma: proto|gpustep:<nl>) systems(comma)
proto   = node-hybrid (Phase 8) B2gpu: ProtoGPU.build(pos) [O(N^2) + nonzero sync] + _Graph.load + replay
gpustep = Phase-9 GPUStep.forces(pos): one graph (NL + model)
Per (mode, system):
  stage times (synced, median of 20): proto: nl, load, replay, step;  gpustep: nl_graph (NL alone captured), step
  torch.profiler (CPU+CUDA) over 3 steps: per-category GPU kernel time per step, GPU busy union, step wall, GPU idle,
  host-sync API calls per step (cudaStreamSynchronize/cudaDeviceSynchronize/cudaEventSynchronize, D2H memcpy).
Kernels are attributed to the NL stage when they start inside the NL window (proto: between the nl CPU range start and
the load range start, which follows the nonzero() sync); gpustep: the NL kernels are the first k kernels of the replay,
k = number of kernels of the NL-only graph (same code captured alone)."""
import os, sys, json, time, re, collections, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/int9")
import common9 as C
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function

out_path, MODES, SYS = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=C.gpuinfo(), runs={})
CATS = [
    ("conv_fwd(node)", r"^void k_[0-9a-f]+p1|nodetp_fwd"),
    ("conv_bwd(node)", r"^kern$|^void kern\b|nodetp_bwd"),
    ("conv(FlashTP)", r"sptp"),
    ("csr_sort/search", r"[Rr]adix|searchsorted|[Oo]nesweep|DeviceSegmentedSort|DeviceMergeSort"),
    ("cueq_SC", r"segmented|cuequivariance|tensor_product|uniform_1d|polynomial|cuet|batch_linear"),
    ("fused_embed", r"fe_(sh|radial)"),
    ("fused_node", r"_seg_linear"),
    ("gemm(radialMLP+linears)", r"gemm|xmma|cutlass|nvjet|cublas|Kernel2|splitK"),
    ("scan/compact", r"[Ss]can|cumsum|nonzero|[Ss]elect|compact"),
    ("scatter/gather/index", r"scatter|gather|index|Index"),
    ("reduce", r"reduce|Reduce"),
    ("elementwise", r"elementwise|vectorized|unrolled|Elementwise"),
    ("memcpy/memset", r"[Mm]emcpy|[Mm]emset"),
]


def cat(n):
    for c, p in CATS:
        if re.search(p, n):
            return c
    return "other"


def union(iv):
    iv = sorted(iv); tot = 0.0; cs = ce = None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None: tot += ce - cs
    return tot


SYNC_API = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


def analyse(p, nsteps, nl_windows=None, nl_first_k=None):
    ev = p.events()
    ANN = ("B2_step", "nl_build", "load", "replay")   # record_function ranges also show up as GPU-side annotations
    gpu = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in ANN]
    gpu.sort(key=lambda e: e.time_range.start)
    cpu = [e for e in ev if e.device_type == torch.autograd.DeviceType.CPU]
    steps = sorted([e for e in cpu if e.name == "B2_step"], key=lambda e: e.time_range.start)
    out = dict(cats=collections.Counter(), top=collections.Counter(), n_kernels=0, nl_us=0.0, nl_kernels=0)
    walls, busy = [], []
    for si, st in enumerate(steps):
        t0, t1 = st.time_range.start, st.time_range.end
        ks = [e for e in gpu if t0 <= e.time_range.start <= t1 + 50]
        walls.append(t1 - t0); busy.append(union([(e.time_range.start, e.time_range.end) for e in ks]))
        nlw = None
        if nl_windows is not None:
            a = [e for e in cpu if e.name == "nl_build" and t0 <= e.time_range.start <= t1]
            b = [e for e in cpu if e.name == "load" and t0 <= e.time_range.start <= t1]
            if a and b: nlw = (a[0].time_range.start, b[0].time_range.start)
        for i, e in enumerate(ks):
            d = e.time_range.end - e.time_range.start
            is_nl = (nlw is not None and nlw[0] <= e.time_range.start < nlw[1]) or (nl_first_k is not None and i < nl_first_k)
            c = "NL_build" if is_nl else cat(e.name)
            if is_nl:
                out["nl_us"] += d; out["nl_kernels"] += 1
            out["cats"][c] += d
            out["top"][(c, e.name[:90])] += d
            out["n_kernels"] += 1
    ns = max(1, len(steps))
    syncs = collections.Counter(e.name for e in cpu if e.name in SYNC_API or (e.name == "cudaMemcpyAsync"))
    r = dict(n_steps=len(steps), wall_us=float(np.median(walls)) if walls else None,
             gpu_busy_us=float(np.median(busy)) if busy else None,
             cats_us_per_step={k: v / ns for k, v in sorted(out["cats"].items(), key=lambda x: -x[1])},
             top_kernels=[(c, n, v / ns) for (c, n), v in out["top"].most_common(30)],
             kernels_per_step=out["n_kernels"] / ns, nl_kernels_per_step=out["nl_kernels"] / ns,
             sync_api_calls_per_step={k: v / ns for k, v in syncs.items()})
    if r["wall_us"]:
        r["gpu_idle_us"] = r["wall_us"] - r["gpu_busy_us"]; r["gpu_idle_frac"] = r["gpu_idle_us"] / r["wall_us"]
    return r


def ev_time(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return C.stats(ts)


built = {}
for mode in MODES:
    base = "nodehybrid+embed+node" if mode == "proto" else "nodeext+embed+node"
    if base not in built:
        built[base] = C.build(base)
    calc, model = built[base]
    for s in SYS:
        atoms = C.load(s); key = f"{mode}/{s}"
        try:
            torch.cuda.empty_cache()
            pos = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
            D = dict(N=len(atoms))
            if mode == "proto":
                rn = C.GR.GraphedRunner(calc, model); rn.step_tensors(atoms)
                k = rn.key_of(atoms); g = rn.graphs[k]; pg = C.ProtoGPU(calc, atoms)
                D["stage_ms"] = dict(nl=ev_time(lambda: pg.build(pos)), replay=ev_time(g.replay))
                bd = pg.build(pos)
                D["stage_ms"]["load"] = ev_time(lambda: g.load(bd))
                D.update(E=int(bd["edge_index"].shape[1]), E_cap=g.E_cap)

                def step():
                    with record_function("B2_step"):
                        with record_function("nl_build"):
                            b = pg.build(pos)
                        with record_function("load"):
                            g.load(b)
                        with record_function("replay"):
                            g.replay()
                        torch.cuda.synchronize()
                D["stage_ms"]["step"] = ev_time(step)
                for _ in range(3): step()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                    for _ in range(3): step()
                D["profile"] = analyse(p, 3, nl_windows=True)
                del rn, g
            else:
                import gpu_step as GS
                nl = mode.split(":")[1] if ":" in mode else "auto"
                sim = GS.GPUStep(calc, model, atoms, nl=nl); sim.forces(pos); D["check"] = sim.check()
                g = sim.graphs[sim.key]
                D.update(E=D["check"]["last_E"], E_cap=g.E_cap, nl=sim.nl_name)
                # NL alone captured (same code, same inputs)
                s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    for _ in range(2): g.NL.build(g.pos)
                torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
                gn = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gn):
                    g.NL.build(g.pos)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as pn:
                    gn.replay(); torch.cuda.synchronize()
                k_nl = len([e for e in pn.events() if e.device_type == torch.autograd.DeviceType.CUDA])
                D["stage_ms"] = dict(nl_graph=ev_time(gn.replay), step=ev_time(lambda: sim.forces(pos)))
                D["nl_graph_kernels"] = k_nl

                def step():
                    with record_function("B2_step"):
                        sim.forces(pos); torch.cuda.synchronize()
                for _ in range(3): step()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                    for _ in range(3): step()
                D["profile"] = analyse(p, 3, nl_first_k=k_nl)
                D["stats"] = {k: v for k, v in sim.stats().items() if k != "events"}
                del sim, g, gn
            res["runs"][key] = D
            pr = D["profile"]
            print(key, json.dumps(dict(stage={k: round(v["median"], 3) for k, v in D["stage_ms"].items()},
                                       wall=pr["wall_us"], busy=pr["gpu_busy_us"], idle=pr.get("gpu_idle_us"),
                                       cats={k: round(v) for k, v in pr["cats_us_per_step"].items()},
                                       syncs=pr["sync_api_calls_per_step"])), flush=True)
        except Exception as ex:  # noqa: BLE001
            res["runs"][key] = dict(error=f"{type(ex).__name__}: {ex}"[:400], tb=traceback.format_exc()[-2500:])
            print(traceback.format_exc(), flush=True)
        json.dump(res, open(out_path, "w"), indent=1, default=str)
print("DONE", flush=True)
