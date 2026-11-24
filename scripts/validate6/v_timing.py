"""validate6 check 5: timing reproduction. usage: python -I v_timing.py OUT.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import *  # noqa
WARM, REPS = 15, 80
out_path = sys.argv[1]
res = header(); res.update(warm=WARM, reps=REPS, systems={})
calcs, builders, models = build_all(("e3nn", "flash", "hybrid"))
r_max = float(calcs["e3nn"].r_max)


def timeit(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), min=float(a.min()))


for sname in ("aspirin", "water192", "water648"):
    at = load(sname); S = res["systems"][sname] = {}
    for bk in ("hybrid", "flash", "e3nn"):
        B = S[bk] = {}
        m, b = models[bk], builders[bk]
        bd = b.build(at); E = bd["edge_index"].shape[1]
        B["E"] = int(E)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        B["eager_model"] = timeit(lambda: eager(m, bd))
        B["eager_full_step"] = timeit(lambda: eager(m, b.build(at)))
        B["eager_peak_MiB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated(); r0 = torch.cuda.memory_reserved()
        t0 = time.perf_counter(); G = GraphedMACE(m, bd, round_cap(E), r_max); torch.cuda.synchronize()
        B["capture_wall_s"] = time.perf_counter() - t0; B["capture_s_reported"] = G.capture_s; B["E_cap"] = G.E_cap
        B["graph_resident_MiB"] = (torch.cuda.memory_allocated() - m0) / 2**20
        B["graph_reserved_delta_MiB"] = (torch.cuda.memory_reserved() - r0) / 2**20
        B["graphed_copyin_replay"] = timeit(lambda: G.step(bd))
        B["graphed_replay_only"] = timeit(lambda: G.replay())
        B["graphed_full_step"] = timeit(lambda: G.step(b.build(at)))
        B["gpu_build_only"] = timeit(lambda: b.build(at))
        B["peak_incl_capture_MiB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
        B["total_peak_process_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        # sanity: graphed result equals eager
        Eg, Fg = G.step(bd); Ee, Fe = eager(m, bd); B["sanity_max_dF"] = float((Fg - Fe).abs().max())
        t0 = time.perf_counter(); G2 = GraphedMACE(m, bd, round_cap(E, 0.25), r_max); torch.cuda.synchronize(); B["recapture_s"] = time.perf_counter() - t0
        del G, G2; torch.cuda.empty_cache()
        print(sname, bk, {k: (v["median"] if isinstance(v, dict) else v) for k, v in B.items()}, flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
res["gpu_end"] = gpuinfo()
json.dump(res, open(out_path, "w"), indent=1)
print("DONE")
