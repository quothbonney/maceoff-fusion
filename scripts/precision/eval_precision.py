"""FlashTP (uvu_TP, fp64) vs e3nn reference on captured real MACE-OFF23 medium conv_tp inputs.
Usage: python -I eval_precision.py <variant_label> <flashtp_root|-> <model> <inputs.pt> <out.json>
  flashtp_root '-' => installed package in venv (unmodified); otherwise prepended to sys.path (worktree)."""
import sys, os, json, time, statistics, hashlib, subprocess
label, ftp_root, model_path, inp_path, out_json = sys.argv[1:6]
if ftp_root != "-":
    sys.path.insert(0, ftp_root)
import torch
import mace  # noqa: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD for e3nn
from e3nn import o3
from mace.calculators import MACECalculator
torch.set_default_dtype(torch.float64)
import flashTP_e3nn

dev = "cuda"
N_REP, N_WARM, N_TIME = 5, 5, 30
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
NAMES1 = ["d_in1", "d_in2", "d_w"]
NAMES2 = ["dd_in1", "dd_in2", "dd_w"]

def err(a, b):
    d = (a - b).abs(); ma = b.abs().max().item()
    rr = b.pow(2).mean().sqrt().item()
    return {"max_abs": d.max().item(), "max_ref": ma, "max_abs_over_max_ref": d.max().item() / ma if ma else None,
            "rms_err_over_rms_ref": (d.pow(2).mean().sqrt().item() / rr) if rr else None}

calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64")
model = calc.models[0]
torch.set_default_dtype(torch.float64)
data = torch.load(inp_path, weights_only=False)

def gather_tp_scatter(tp, x, y, w, src, dst, n):
    mji = tp(x[src], y, w)
    return torch.zeros(n, mji.shape[1], dtype=mji.dtype, device=mji.device).index_add_(0, dst, mji)

def dense_emul(tp, cg_fp32):
    """Dense re-statement of the uvu TP (e3nn semantics) with optional fp32-rounded CG."""
    i1, i2, io = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
    s1, s2, so = i1.slices(), i2.slices(), io.slices()
    paths, off = [], 0
    for ins in tp.instructions:
        mu, ir1 = i1[ins.i_in1]; mv, ir2 = i2[ins.i_in2]; mo, ir3 = io[ins.i_out]
        assert mv == 1 and ins.connection_mode == "uvu"
        cg = o3.wigner_3j(ir1.l, ir2.l, ir3.l, dtype=torch.float64)
        if cg_fp32: cg = cg.to(torch.float32).to(torch.float64)
        paths.append((ins, mu, ir1.dim, ir3.dim, off, cg.to(dev)))
        off += mu * mv
    def f(x, y, w, src, dst, n):
        xs = x[src]; out = torch.zeros(n, io.dim, dtype=x.dtype, device=x.device)
        parts = [None] * len(io)
        for ins, mu, d1, d3, o_, cg in paths:
            z = ins.path_weight * torch.einsum("eui,ej,ijk,eu->euk", xs[:, s1[ins.i_in1]].reshape(-1, mu, d1),
                                               y[:, s2[ins.i_in2]], cg, w[:, o_:o_ + mu])
            z = z.reshape(z.shape[0], -1)
            parts[ins.i_out] = z if parts[ins.i_out] is None else parts[ins.i_out] + z
        mji = torch.cat([p if p is not None else torch.zeros(xs.shape[0], m * ir.dim, device=dev) for p, (m, ir) in zip(parts, io)], 1)
        return out.index_add_(0, dst, mji)
    return f

def run_all(fn, x, y, w, src, dst, n, C, C2, second=True):
    xs, ys, ws = [t.detach().clone().requires_grad_(True) for t in (x, y, w)]
    out = fn(xs, ys, ws, src, dst, n)
    g = torch.autograd.grad((out * C).sum(), [xs, ys, ws], create_graph=second)
    r = {"out": out.detach()}
    r.update({k: v.detach() for k, v in zip(NAMES1, g)})
    if second:
        L2 = sum((gi * ci).sum() for gi, ci in zip(g, C2))
        gg = torch.autograd.grad(L2, [xs, ys, ws], allow_unused=True)
        r.update({k: (v.detach() if v is not None else torch.zeros_like(t)) for k, v, t in zip(NAMES2, gg, (x, y, w))})
    return r

def timeit(f):
    for _ in range(N_WARM): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(N_TIME):
        torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return {"median_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3, "n": N_TIME, "warmup": N_WARM}

res = {"label": label, "flashtp_file": flashTP_e3nn.__file__,
       "flashtp_git": None, "gpu": torch.cuda.get_device_name(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
       "torch": torch.__version__, "inputs": inp_path, "cotangent": "fixed normal(seed=1234) C for L=sum(out*C); "
       "second order: L2=sum_k sum(grad_k * C2_k), C2 seed 4321", "systems": {}}
if ftp_root != "-":
    res["flashtp_git"] = subprocess.run(["git", "-C", ftp_root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

for li, inter in enumerate(model.interactions):
    tp = inter.conv_tp.to(dev)
    insts = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]
    ftp = flashTP_e3nn.uvu_TP(tp.irreps_in1, tp.irreps_in2, tp.irreps_out, insts, device=dev, dtype=torch.float64)
    # CG table accuracy vs exact fp64 wigner_3j
    tab = torch.tensor(ftp.unique_cg_val, dtype=torch.float64)
    worst = 0.0
    for ins in tp.instructions:
        cg = o3.wigner_3j(tp.irreps_in1[ins.i_in1].ir.l, tp.irreps_in2[ins.i_in2].ir.l, tp.irreps_out[ins.i_out].ir.l, dtype=torch.float64)
        for v in cg[cg != 0].tolist():
            worst = max(worst, (tab - v).abs().min().item())
    res.setdefault("cg_table", {})[f"layer{li}"] = {"n_unique": len(ftp.unique_cg_val), "kernel_class": type(ftp).__name__,
                                                  "max_abs_table_minus_exact_cg": worst}
    log(f"layer{li}: {type(ftp).__name__} n_unique_cg={len(ftp.unique_cg_val)} worst_cg_err={worst:.3e}")
    emu64, emu32 = dense_emul(tp, False), dense_emul(tp, True)
    def ftp_fn(x, y, w, src, dst, n, _s={}):
        return ftp(x, y, w, src.int().contiguous(), dst.int().contiguous())
    for sname, sysd in data["systems"].items():
        c = sysd["layers"][li]
        x = c["node_feats"].to(dev); y = c["edge_attrs"].to(dev); w = c["tp_weights"].to(dev)
        src, dst = c["edge_index"][0].to(dev), c["edge_index"][1].to(dev); n = x.shape[0]
        src32, dst32 = src.int().contiguous(), dst.int().contiguous()
        g = torch.Generator(device="cpu").manual_seed(1234)
        C = torch.randn(n, tp.irreps_out.dim, generator=g, dtype=torch.float64).to(dev)
        g2 = torch.Generator(device="cpu").manual_seed(4321)
        C2 = [torch.randn(t.shape, generator=g2, dtype=torch.float64).to(dev) for t in (x, y, w)]
        ref = run_all(lambda *a: gather_tp_scatter(tp, *a), x, y, w, src, dst, n, C, C2)
        e64 = run_all(emu64, x, y, w, src, dst, n, C, C2)
        e32 = run_all(emu32, x, y, w, src, dst, n, C, C2)
        reps = [run_all(lambda x_, y_, w_, s_, d_, n_: ftp(x_, y_, w_, src32, dst32), x, y, w, src, dst, n, C, C2)
                for _ in range(N_REP)]
        fl = reps[0]
        keys = list(ref)
        entry = {"n_nodes": n, "n_edges": int(src.numel()),
                 "flashtp_vs_e3nn": {k: err(fl[k], ref[k]) for k in keys},
                 "cgfp32_emul_vs_e3nn": {k: err(e32[k], ref[k]) for k in keys},
                 "flashtp_vs_cgfp32_emul": {k: err(fl[k], e32[k]) for k in keys},
                 "flashtp_vs_exactcg_emul": {k: err(fl[k], e64[k]) for k in keys},
                 "exactcg_emul_vs_e3nn(sanity)": {k: err(e64[k], ref[k]) for k in keys},
                 "repeat": {k: {"bitwise_identical_all": all(torch.equal(r[k], fl[k]) for r in reps[1:]),
                                "max_abs_diff_vs_rep0": max((r[k] - fl[k]).abs().max().item() for r in reps[1:]),
                                "max_abs_diff_over_max_ref": max((r[k] - fl[k]).abs().max().item() for r in reps[1:]) / ref[k].abs().max().item()}
                            for k in keys}}
        # reference determinism (e3nn + index_add on GPU also uses atomics)
        ref2 = run_all(lambda *a: gather_tp_scatter(tp, *a), x, y, w, src, dst, n, C, C2, second=False)
        entry["e3nn_ref_repeat_max_abs_diff"] = {k: (ref2[k] - ref[k]).abs().max().item() for k in ["out"] + NAMES1}
        if sname == "water648":
            xs, ys, ws = [t.clone().requires_grad_(True) for t in (x, y, w)]
            def fwd(): 
                with torch.no_grad(): ftp(x, y, w, src32, dst32)
            def fwdbwd():
                o = ftp(xs, ys, ws, src32, dst32); torch.autograd.grad(o, [xs, ys, ws], C)
            def rfwd():
                with torch.no_grad(): gather_tp_scatter(tp, x, y, w, src, dst, n)
            def rfwdbwd():
                o = gather_tp_scatter(tp, xs, ys, ws, src, dst, n); torch.autograd.grad(o, [xs, ys, ws], C)
            entry["timing"] = {"flashtp_fwd": timeit(fwd), "flashtp_fwd_bwd": timeit(fwdbwd),
                               "e3nn_fwd": timeit(rfwd), "e3nn_fwd_bwd": timeit(rfwdbwd)}
        res["systems"].setdefault(sname, {})[f"layer{li}"] = entry
        log(sname, f"layer{li}", "out", entry["flashtp_vs_e3nn"]["out"], "rep_identical", entry["repeat"]["out"]["bitwise_identical_all"],
            entry.get("timing", {}).get("flashtp_fwd"))
        del ref, e64, e32, reps, fl; torch.cuda.empty_cache()
json.dump(res, open(out_json, "w"), indent=1)
log("wrote", out_json)
