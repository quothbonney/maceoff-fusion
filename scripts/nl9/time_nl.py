"""nl9 builder microbenchmark (B-level): cell-list builder (eager launches; CUDA-graph replay) vs proto_graph build
(incl. its H2D + nonzero host sync) vs stock get_neighborhood (matscipy CPU). usage: python -I time_nl.py OUT.json"""
import os, sys, json, time, subprocess
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/nl9")
import numpy as np
import torch
import nlsys
from nlsys import load, stock_nl
from cell_nl import CellNeighborList, round_cap
import proto_graph
from mace.calculators import MACECalculator

OUT = sys.argv[1]
WARM, REPS = 10, int(os.environ.get("REPS", 100))
dev = torch.device("cuda")
pc = time.perf_counter
sync = torch.cuda.synchronize


def gpuinfo():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def stats(x):
    x = np.asarray(x) * 1e3
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return dict(median_ms=float(p50), p10_ms=float(p10), p90_ms=float(p90), n=int(x.size))


def bench(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    sync()
    t = []
    for _ in range(reps):
        t0 = pc(); fn(); sync(); t.append(pc() - t0)
    return stats(t)


squeue = subprocess.run(["squeue", "-h", "-o", "%i %j %T"], capture_output=True, text=True).stdout.strip()
res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), torch=torch.__version__,
           warmup=WARM, reps=REPS, squeue_at_start=squeue, rows=[])
calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64")
r_max = float(calc.r_max)
pb = proto_graph.ProtoBuilder(calc)

SYS = ["aspirin", "water192", "org648", "water648", "wbox648", "wbox648_tric", "wclus2187", "wbox2187", "wclus5184",
       "wbox5184"]
for name in SYS:
    a = load(name)
    N = len(a)
    ei, *_ = stock_nl(a, r_max)
    E = int(ei.shape[1])
    pbc = tuple(bool(x) for x in a.pbc)
    E_cap = round_cap(1.2 * E + 1, 64)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    nl = CellNeighborList(r_max, pbc, np.array(a.cell), N, E_cap, "cuda")
    pos = torch.tensor(a.positions, device=dev)
    nl.build(pos); sync()
    m_nl = torch.cuda.max_memory_allocated() - m0
    row = dict(system=name, N=N, E=E, E_cap=E_cap, pbc=any(pbc), nbins=nl.nbins, bin_cap=nl.bin_cap,
               nl_buffers_bytes=nl.memory_bytes(), nl_peak_alloc_bytes=int(m_nl))
    # cell-list eager (launches + sync)
    row["cell_eager"] = bench(lambda: nl.build(pos))
    # cell-list graph replay (builder only)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        nl.build(pos); nl.build(pos)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        nl.build(pos)
    row["cell_graph"] = bench(g.replay)
    # device time via events (graph replay)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dt = []
    for _ in range(REPS):
        ev0.record(); g.replay(); ev1.record(); ev1.synchronize(); dt.append(ev0.elapsed_time(ev1) / 1e3)
    row["cell_graph_device"] = stats(dt)
    # per-kernel breakdown (eager launches, 20 reps)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            nl.build(pos)
        sync()
    kt = {}
    for ev in prof.key_averages():
        if ev.device_type.name == "CUDA" or getattr(ev, "self_device_time_total", 0) > 0:
            kt[ev.key[:40]] = getattr(ev, "self_device_time_total", 0) / 20 / 1e3   # ms per build
    row["kernels_ms"] = {k: v for k, v in sorted(kt.items(), key=lambda x: -x[1]) if v > 0}
    assert int(nl.out["overflow"].item()) == 0 and int(nl.out["n_edges"].item()) == E
    del g
    # proto_graph build (host positions -> H2D -> O(N^2) -> nonzero sync)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    pb.build(a); sync()
    row["proto_peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated() - m0)
    reps_p = REPS if N < 3000 else 30
    row["proto_build"] = bench(lambda: pb.build(a), reps=reps_p)
    # stock CPU (matscipy via get_neighborhood)
    reps_s = REPS if N < 3000 else 30
    row["stock_get_neighborhood_cpu"] = bench(lambda: stock_nl(a, r_max), reps=reps_s, warm=3)
    res["rows"].append(row)
    print(f"{name:14s} N={N:5d} E={E:7d} cell eager {row['cell_eager']['median_ms']:.3f} graph "
          f"{row['cell_graph']['median_ms']:.3f} (dev {row['cell_graph_device']['median_ms']:.3f}) proto "
          f"{row['proto_build']['median_ms']:.3f} matscipy {row['stock_get_neighborhood_cpu']['median_ms']:.2f} ms; "
          f"nl buf {row['nl_buffers_bytes']/2**20:.2f} MiB proto peak {row['proto_peak_alloc_bytes']/2**20:.1f} MiB",
          flush=True)
    print("   kernels ms/build:", {k: round(v, 4) for k, v in row["kernels_ms"].items()}, flush=True)
    del nl
res["gpu_end"] = gpuinfo()
res["squeue_at_end"] = subprocess.run(["squeue", "-h", "-o", "%i %j %T"], capture_output=True, text=True).stdout.strip()
json.dump(res, open(OUT, "w"), indent=1)
