"""prof8 conv-only benchmark on captured real MACE-OFF23 medium inputs (results/prof8/inputs_<sys>.pt, layer -> [x,y,w,ei]).
usage: python -I conv8.py OUT.json systems impls   (impls comma: flash,e3nn,node[,oeq,luna])
Per (system, layer, impl): local correctness vs e3nn (out, gx, gy, gw; max abs and max abs / max|ref|),
eager fwd / bwd / fwd+bwd synced wall (>=10 warm, 60 reps; median p10 p90), CUDA-graph replay of fwd and fwd+bwd,
GPU kernel time per fwd+bwd (torch.profiler), kernel count, peak memory. node = node-centric conv incl. CSR build every call
(csr_cache off: conservative, in the model the CSR is built once for both layers); csr build timed separately.
"""
import os, sys, json, time, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HG  # FIRST
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity
from mace.tools.scatter import scatter_sum

out_path = sys.argv[1]; SYS = sys.argv[2].split(","); IMPL = sys.argv[3].split(",")
WARM, REPS = 10, 60
calc, _ = HG.build_calc(conv="flashtp", sc="e3nn", check_indices=False)
flash = [b.conv_tp for b in calc.models[0].interactions]
e3 = [f.orig for f in flash]
impls = {"flash": lambda i: flash[i],
         "e3nn": lambda i: (lambda x, y, w, ei: scatter_sum(e3[i](x[ei[0]], y, w), ei[1], dim=0, dim_size=x.shape[0]))}
if "node" in IMPL:
    sys.path.insert(0, f"{R}/scripts/nodetp8_bwd"); import nodeconv as NC
    from csr import build_csr
    _fwd_fn, _fwd_name = NC._load_fwd(os.environ.get("NODE_FWD", "auto"))
    nodes = [NC.NodeConv(e3[i], _fwd_fn, _fwd_name, NC._CsrCache(), csr_cache=False) for i in range(2)]
    impls["node"] = lambda i: nodes[i]
if "oeq" in IMPL or "luna" in IMPL:
    sys.path.insert(0, f"{R}/scripts/prof8"); import extra8  # optional adapters (written only if A reports them working)
    for k in ("oeq", "luna"):
        if k in IMPL: impls[k] = extra8.make(k, e3)


def stats(ts):
    a = np.array(ts); return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), n=len(ts))


def timeit(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return stats(ts)


def gpuinfo():
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


res = dict(job=os.environ.get("SLURM_JOB_ID"), gpu_start=gpuinfo(), warm=WARM, reps=REPS, node_fwd=globals().get("_fwd_name"), data={})
for s in SYS:
    inp = torch.load(f"{R}/results/prof8/inputs_{s}.pt")
    for i in range(2):
        x0, y0, w0, ei = [t.cuda() for t in inp[i]]
        N, E = x0.shape[0], ei.shape[1]
        g = torch.randn(N, flash[i].out_dim, dtype=torch.float64, device="cuda", generator=torch.Generator("cuda").manual_seed(i))
        leaves = [t.clone().requires_grad_(True) for t in (x0, y0, w0)]
        ref_o = impls["e3nn"](i)(*leaves, ei); ref_g = torch.autograd.grad(ref_o, leaves, g)
        ref = [ref_o.detach()] + list(ref_g)
        D = res["data"][f"{s}/L{i}"] = dict(N=N, E=E)
        if "node" in IMPL:
            D["csr_build"] = timeit(lambda: build_csr(ei, N))
        for k in IMPL:
            try:
                f = impls[k](i)
                xs = [t.clone().requires_grad_(True) for t in (x0, y0, w0)]
                o = f(*xs, ei); gs = torch.autograd.grad(o, xs, g)
                err = {}
                for nm, a, b in zip(("out", "gx", "gy", "gw"), [o.detach()] + list(gs), ref):
                    d = (a - b).abs().max().item(); m = b.abs().max().item()
                    nz = b.abs() < 1e-8
                    err[nm] = dict(max_abs=d, rel=d / max(m, 1e-300), near_zero_max_abs=(a - b).abs()[nz].max().item() if nz.any() else 0.0)
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
                def fwd():
                    with torch.no_grad(): return f(x0, y0, w0, ei)
                def fb():
                    o = f(*xs, ei); return torch.autograd.grad(o, xs, g)
                T = dict(err=err)
                T["eager_fwd"] = timeit(fwd); T["eager_fwdbwd"] = timeit(fb)
                o = f(*xs, ei)
                T["eager_bwd"] = timeit(lambda: torch.autograd.grad(o, xs, g, retain_graph=True))
                T["peak_MiB_fwdbwd"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
                del o
                # CUDA graphs
                st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    for _ in range(3): fwd(); fb()
                torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
                gf = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gf): fwd()
                gfb = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gfb): fb()
                T["graph_fwd"] = timeit(gf.replay); T["graph_fwdbwd"] = timeit(gfb.replay)
                T["graph_bwd_est"] = T["graph_fwdbwd"]["median"] - T["graph_fwd"]["median"]
                for nm, fn in (("fwd", gf.replay), ("fwdbwd", gfb.replay)):
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as p:
                        for _ in range(5): fn()
                        torch.cuda.synchronize()
                    ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    T[f"gpu_us_{nm}"] = sum((getattr(e, "device_time", 0) or e.cuda_time) for e in ks) / 5
                    T[f"n_gpu_ops_{nm}"] = len(ks) / 5
                    if nm == "fwdbwd":
                        agg = {}
                        for e in ks: agg[e.name[:70]] = agg.get(e.name[:70], 0) + (getattr(e, "device_time", 0) or e.cuda_time) / 5
                        T["top_kernels_fwdbwd"] = dict(sorted(agg.items(), key=lambda t: -t[1])[:6])
                del gf, gfb
                D[k] = T
                print(s, i, k, json.dumps(dict(rel=max(v["rel"] for v in err.values()), ef=round(T["eager_fwd"]["median"], 4),
                      efb=round(T["eager_fwdbwd"]["median"], 4), gf=round(T["graph_fwd"]["median"], 4),
                      gfb=round(T["graph_fwdbwd"]["median"], 4), gpu_fb_us=round(T["gpu_us_fwdbwd"], 1), nk=T["n_gpu_ops_fwdbwd"],
                      peak=round(T["peak_MiB_fwdbwd"]))), flush=True)
            except Exception as ex:
                D[k] = dict(error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc()); print(D[k]["tb"], flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
res["gpu_end"] = gpuinfo()
json.dump(res, open(out_path, "w"), indent=1)
print("DONE", flush=True)
