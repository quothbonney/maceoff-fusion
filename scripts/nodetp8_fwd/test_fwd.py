"""Correctness + timing of node-centric fwd vs reference (MACE conv_tp + gather + scatter_sum), FlashTP-graphcap, e3nn.
usage: python -I test_fwd.py OUT.json [systems(comma)] [layers(comma)] [--notime]"""
import os, sys, json, time
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc  # noqa: F401  FIRST (FlashTP cgpatch import order)
import flashtp_adapter_gc as FA
sys.path.insert(0, f"{R}/scripts/nodetp8_fwd")
import numpy as np
import torch
import ase.io
from e3nn import o3
from mace.calculators import MACECalculator
from mace.tools.scatter import scatter_sum
import nodetp_fwd as NT

out_path = sys.argv[1]
SYS = (sys.argv[2] if len(sys.argv) > 2 else "aspirin,org192,water192,org648,water648,wbox648").split(",")
LAYERS = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")]
DOTIME = "--notime" not in sys.argv
WARM, REPS = 10, 50
dev = torch.device("cuda")

try:
    sys.path.insert(0, f"{R}/scripts/nodetp8_bwd")
    import csr as CSRMOD  # C's canonical
    build_csr, csr_src = CSRMOD.build_csr, "nodetp8_bwd/csr.py"
except Exception as ex:  # noqa
    build_csr, csr_src = NT.build_csr, f"local stub ({type(ex).__name__})"


def stats(ts):
    a = np.array(ts) * 1e3
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), n=len(a))


def timeit(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return stats(ts)


def graph_time(fn, k=1):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k): fn()
    r = timeit(g.replay)
    if k > 1:
        r = {kk: (v / k if kk != "n" else v) for kk, v in r.items()}
        r["calls_per_graph"] = k
    return r


def gpuinfo():
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def err(o, r):
    d = (o - r).abs()
    m = float(r.abs().max()) if r.numel() else 0.0
    nz = r.abs() < 1e-8
    return dict(max_abs=float(d.max()) if d.numel() else 0.0, rel_to_max=(float(d.max()) / m if m > 0 else 0.0),
                max_ref=m, n_near_zero=int(nz.sum()), max_abs_near_zero=float(d[nz].max()) if nz.any() else 0.0,
                n_nonfinite=int((~torch.isfinite(o)).sum()))


calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64")
model = calc.models[0]
torch.set_default_dtype(torch.float64)
cap = {}
for li, inter in enumerate(model.interactions):
    inter.register_forward_pre_hook(lambda m, a, kw, li=li: cap.setdefault(li, {}).update(
        edge_index=kw["edge_index"].detach().clone(), y=kw["edge_attrs"].detach().clone()), with_kwargs=True)
    inter.linear_up.register_forward_hook(lambda m, a, o, li=li: cap[li].update(x=o.detach().clone()))
    inter.conv_tp.register_forward_pre_hook(lambda m, a, li=li: cap[li].update(w=a[2].detach().clone()))

res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), csr=csr_src,
           warm=WARM, reps=REPS, layers={}, systems={})
cfgs, flash = {}, {}
for li in LAYERS:
    tp = model.interactions[li].conv_tp
    cfgs[li] = NT.make_cfg(tp)
    t0 = time.perf_counter(); NT.get_ext(cfgs[li]); tb = time.perf_counter() - t0
    # CG check: module w3j buffers vs o3.wigner_3j fp64
    w3 = {n.split("._w3j_")[-1]: b for n, b in tp.named_buffers() if "_w3j_" in n}
    cgd = max([float((b - o3.wigner_3j(*[int(c) for c in k.split("_")], dtype=torch.float64).to(b)).abs().max())
               for k, b in w3.items()] or [0.0])
    flash[li] = FA.FlashConv(tp, torch.float64, dev, check_indices=False)
    res["layers"][li] = dict(nnz=cfgs[li].nnz, flops_per_edge=cfgs[li].flops_per_edge(), W=cfgs[li].W,
                             d1=cfgs[li].d1, d_out=cfgs[li].d_out, ext_build_s=tb, max_module_w3j_vs_o3_fp64=cgd,
                             paths=[dict(l=p["l"], path_weight=p["path_weight"], nnz=len(p["cg"])) for p in cfgs[li].paths])
    print("layer", li, res["layers"][li], flush=True)


def ref_conv(tp, x, y, w, ei):
    return scatter_sum(src=tp(x[ei[0]], y, w), index=ei[1], dim=0, dim_size=x.shape[0])


def run_case(tag, li, x, y, w, ei, timing):
    tp = model.interactions[li].conv_tp
    N, E = x.shape[0], ei.shape[1]
    with torch.no_grad():
        ref = ref_conv(tp, x, y, w, ei)
        csr = build_csr(ei, N)
        out = NT.fwd(cfgs[li], x, y, w, ei, csr)
        torch.cuda.synchronize()
        r = dict(N=N, E=E, G=NT.choose_G(N, E, cfgs[li]), PG=NT.default_PG(cfgs[li]), err=err(out, ref))
        if E > 0:
            fo = flash[li](x, y, w, ei)
            r["flash_vs_ref"] = err(fo, ref)
        if timing:
            T = {}
            T["csr_build"] = timeit(lambda: build_csr(ei, N))
            T["csr_build_graph"] = graph_time(lambda: build_csr(ei, N))
            o = torch.empty_like(out)
            T["node_fwd"] = timeit(lambda: NT.fwd(cfgs[li], x, y, w, ei, csr, out=o))
            T["node_fwd_graph"] = graph_time(lambda: NT.fwd(cfgs[li], x, y, w, ei, csr, out=o))
            T["node_fwd_graph_x20"] = graph_time(lambda: NT.fwd(cfgs[li], x, y, w, ei, csr, out=o), k=20)
            T["node_csr_fwd_graph"] = graph_time(lambda: NT.fwd(cfgs[li], x, y, w, ei, build_csr(ei, N)))
            if E > 0:
                T["flashtp_fwd"] = timeit(lambda: flash[li](x, y, w, ei))
                T["flashtp_fwd_graph"] = graph_time(lambda: flash[li](x, y, w, ei))
                T["flashtp_fwd_graph_x20"] = graph_time(lambda: flash[li](x, y, w, ei), k=20)
            T["e3nn_fwd"] = timeit(lambda: ref_conv(tp, x, y, w, ei))
            T["e3nn_fwd_graph"] = graph_time(lambda: ref_conv(tp, x, y, w, ei))
            c = cfgs[li]
            t = T["node_fwd_graph_x20"]["median"] * 1e-3  # s
            byt = 8 * (E * (c.W + c.d2 + c.d1) + N * c.d_out) + 4 * (E * 3 + N)  # w,y,x-gather (upper), out, idx
            byt_min = 8 * (E * (c.W + c.d2) + N * (c.d1 + c.d_out))
            r["achieved"] = dict(GBps_gather_model=byt / t / 1e9, GBps_unique_bytes=byt_min / t / 1e9,
                                 TFLOPs=c.flops_per_edge() * E / t / 1e12, bytes_unique=byt_min,
                                 flops=c.flops_per_edge() * E,
                                 roofline_us_hbm=byt_min / 3.35e12 * 1e6, roofline_us_fp64=c.flops_per_edge() * E / 33.5e12 * 1e6)
            r["timing_ms"] = T
        return r


for s in SYS:
    atoms = ase.io.read(f"{R}/results/workloads/systems/{s}.extxyz")
    cap.clear(); atoms.calc = calc
    e = atoms.get_potential_energy()
    res["systems"][s] = dict(n_atoms=len(atoms), energy=float(e), layers={})
    for li in LAYERS:
        c = cap[li]
        ei = c["edge_index"]
        srt = bool((ei[0][1:] >= ei[0][:-1]).all()) if ei.shape[1] > 1 else True
        r = run_case(s, li, c["x"], c["y"], c["w"], ei, DOTIME)
        r["edges_sorted_by_src"] = srt
        res["systems"][s]["layers"][li] = r
        print(s, li, json.dumps(r)[:1500], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

# synthetic: E=0, and random with padded self-edges (zero weights), isolated nodes
g = torch.Generator(device="cuda").manual_seed(0)
syn = {}
for li in LAYERS:
    c = cfgs[li]
    for name, N, E, pad in (("E0_N5", 5, 0, 0), ("E0_N1", 1, 0, 0), ("rand_N300_E5000_pad700", 300, 5000, 700),
                            ("rand_N7_E3_pad4", 7, 3, 4)):
        x = torch.randn(N, c.d1, generator=g, device=dev)
        y = torch.randn(E + pad, c.d2, generator=g, device=dev)
        w = torch.randn(E + pad, c.W, generator=g, device=dev)
        srcs = torch.randint(0, max(N - 1, 1), (E,), generator=g, device=dev)  # last node never src/dst -> zero row
        dsts = torch.randint(0, max(N - 1, 1), (E,), generator=g, device=dev)
        ps = torch.randint(0, N, (pad,), generator=g, device=dev)
        ei = torch.stack([torch.cat([srcs, ps]), torch.cat([dsts, ps])])
        if pad: w[E:] = 0.0
        o = torch.argsort(ei[0], stable=True); ei = ei[:, o].contiguous(); y = y[o].contiguous(); w = w[o].contiguous()
        r = run_case(name, li, x, y, w, ei, False)
        syn[f"{name}_L{li}"] = r
        print(name, li, json.dumps(r), flush=True)
res["synthetic"] = syn
res["gpu_end"] = gpuinfo()
json.dump(res, open(out_path, "w"), indent=1)
print("DONE", flush=True)
