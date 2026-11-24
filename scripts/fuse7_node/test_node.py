"""fuse7_node local tests (GPU). usage: python -I test_node.py OUT.json"""
import os, sys, json, time, math, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HB  # first: pins flashTP_e3nn
sys.path.insert(0, f"{R}/scripts/fuse7_node")
import numpy as np
import torch
import ase.io, ase.build
from ase import Atoms
from mace.tools import torch_tools
import fused_node as FN

out_path = sys.argv[1]
dev = torch.device("cuda")
RES = dict(job=os.environ.get("SLURM_JOB_ID"), describe=FN.describe(), op={}, model={}, graph={}, bench={}, misc={})


def save():
    json.dump(RES, open(out_path, "w"), indent=1, default=str)


def load(name):
    if name == "H2O":
        return ase.build.molecule("H2O")
    if name == "single_C":
        return Atoms("C", positions=[[0, 0, 0]])
    if name == "far_OO":
        return Atoms("OO", positions=[[0, 0, 0], [20.0, 0, 0]])
    return ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")


def batch_dict(calc, atoms):
    b = calc._atoms_to_batch(atoms)
    return calc._clone_batch(b).to_dict()


def run_model(model, bd, stress):
    bd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bd.items()}
    with torch_tools.default_dtype("float64"):
        o = model(bd, compute_force=True, compute_stress=stress, training=False)
    return (o["energy"].detach().clone(), o["forces"].detach().clone(),
            o["stress"].detach().clone() if stress and o.get("stress") is not None else None)


def cmp_model(a, b, n):
    E, F, S = a; Er, Fr, Sr = b
    d = (F - Fr).abs()
    r = dict(dE_per_atom=float((E - Er).abs().max()) / max(n, 1), max_dF=float(d.max()) if d.numel() else 0.0,
             rms_dF=float(((F - Fr) ** 2).sum(-1).mean().sqrt()) if d.numel() else 0.0)
    if S is not None and Sr is not None:
        r["max_dstress"] = float((S - Sr).abs().max())
    return r


def errs(a, b):
    d = (a - b).abs()
    return dict(max_abs=float(d.max()) if d.numel() else 0.0,
                rel=float(d.max() / b.abs().max().clamp_min(1e-300)) if d.numel() else 0.0)


# ---------------------------------------------------------------------- 0. Triton fp64 dot sanity
try:
    import triton, triton.language as tl

    @triton.jit
    def _dk(A, B, C):
        i = tl.arange(0, 32)
        j = tl.arange(0, 64)
        a = tl.load(A + i[:, None] * 32 + i[None, :])
        b = tl.load(B + i[:, None] * 64 + j[None, :])
        tl.store(C + i[:, None] * 64 + j[None, :], tl.dot(a, b))
    A = torch.randn(32, 32, dtype=torch.float64, device=dev); B = torch.randn(32, 64, dtype=torch.float64, device=dev)
    C = torch.empty(32, 64, dtype=torch.float64, device=dev)
    _dk[(1,)](A, B, C)
    ref = (A.cpu() @ B.cpu()).to(dev)
    RES["misc"]["triton_fp64_dot"] = errs(C, ref)
    try:
        asm = list(_dk.cache[0].values())[0].asm["ptx"] if hasattr(_dk, "cache") else ""
    except Exception:
        asm = ""
    RES["misc"]["fp64_dot_ptx_has_mma_f64"] = ("mma.sync.aligned.m8n8k4.row.col.f64" in asm) or ("f64.f64.f64.f64" in asm)
    RES["misc"]["fp64_dot_ptx_has_fma_f64"] = "fma.rn.f64" in asm
except Exception as e:
    RES["misc"]["triton_fp64_dot"] = f"FAIL {e!r}"
print("misc", RES["misc"], flush=True)

# ---------------------------------------------------------------------- 1. build stock e3nn model, capture inputs
calc, info = HB.build_calc(conv="e3nn", sc="e3nn", dtype=torch.float64)
model = calc.models[0]
cap = {}


def hook_block(i):
    def h(mod, args, kwargs):
        cap.setdefault(i, {})["node_attrs"] = kwargs["node_attrs"].detach().clone()
        cap[i]["node_feats"] = kwargs["node_feats"].detach().clone()
    return h


def hook_lin(i):
    def h(mod, args):
        cap.setdefault(i, {})["message"] = args[0].detach().clone()
    return h


SYSTEMS = ["aspirin", "water648", "wbox648"]
inputs = {}
for nm in SYSTEMS:
    at = load(nm)
    hs = [b.register_forward_pre_hook(hook_block(i), with_kwargs=True) for i, b in enumerate(model.interactions)]
    hs += [b.linear.register_forward_pre_hook(hook_lin(i)) for i, b in enumerate(model.interactions)]
    cap.clear()
    run_model(model, batch_dict(calc, at), False)
    for h in hs:
        h.remove()
    inputs[nm] = {i: dict(v) for i, v in cap.items()}
    print(nm, {i: {k: tuple(t.shape) for k, t in v.items()} for i, v in inputs[nm].items()}, flush=True)

# ---------------------------------------------------------------------- 2. op-level parity
t0 = time.time()
handle = FN.install(model)
RES["misc"]["install_s"] = time.time() - t0
RES["misc"]["install_info"] = handle["info"]
print("install", RES["misc"]["install_s"], handle["info"], flush=True)
blocks = model.interactions
avg = [b.avg_num_neighbors for b in blocks]


def ref_ops(i, x, attrs):
    b = blocks[i]
    if i == 0:
        return {"out": b.reshape(b.skip_tp(b.linear(x) / b.avg_num_neighbors, attrs))}
    return {"out": b.reshape(b.linear(x) / b.avg_num_neighbors)}


def ref_sc(i, nf, attrs):
    return blocks[i].skip_tp(nf, attrs)


def op_case(tag, i, which, x, attrs):
    ops = blocks[i].__dict__[FN._OPS]
    x1 = x.clone().requires_grad_(True); x2 = x.clone().requires_grad_(True)
    if which == "out":
        ref = ref_ops(i, x1, attrs)["out"]
    else:
        ref = ref_sc(i, x1, attrs)
    fu = ops[which](x2, attrs)
    g = torch.randn_like(ref)
    gr, = torch.autograd.grad(ref, x1, g)
    gf, = torch.autograd.grad(fu, x2, g)
    r = dict(N=x.shape[0], shape=list(fu.shape), fwd=errs(fu, ref), bwd=errs(gf, gr))
    RES["op"][f"{tag}/L{i}.{which}"] = r
    return r


worst = dict(fwd=0.0, bwd=0.0)
for nm in SYSTEMS:
    for i in (0, 1):
        c = inputs[nm][i]
        for which in (["out"] if i == 0 else ["out", "sc"]):
            x = c["message"] if which == "out" else c["node_feats"]
            for n in (None, 1, 2, 3, 0):
                if n is not None and nm != "aspirin":
                    continue
                xs = x if n is None else x[:n]
                at_ = c["node_attrs"] if n is None else c["node_attrs"][:n]
                tag = nm if n is None else f"{nm}[:{n}]"
                try:
                    r = op_case(tag, i, which, xs, at_)
                    worst["fwd"] = max(worst["fwd"], r["fwd"]["rel"]); worst["bwd"] = max(worst["bwd"], r["bwd"]["rel"])
                    print(tag, i, which, r, flush=True)
                except Exception as e:
                    RES["op"][f"{tag}/L{i}.{which}"] = f"FAIL {traceback.format_exc()}"
                    print("FAIL", tag, i, which, traceback.format_exc(), flush=True)
RES["misc"]["op_worst_rel"] = worst
# general (non one-hot) attrs on L0 and L1.sc
c = inputs["aspirin"]
ga = torch.rand_like(c[0]["node_attrs"])
RES["op"]["general_attrs/L0.out"] = op_case("general_attrs", 0, "out", c[0]["message"], ga)
RES["op"]["general_attrs/L1.sc"] = op_case("general_attrs", 1, "sc", c[1]["node_feats"], ga)
print("general attrs", RES["op"]["general_attrs/L0.out"], RES["op"]["general_attrs/L1.sc"], flush=True)
save()

# gradcheck tiny
try:
    for i, which in ((0, "out"), (1, "out"), (1, "sc")):
        ci = inputs["aspirin"][i]
        x = (ci["message"] if which == "out" else ci["node_feats"])[:2].clone().requires_grad_(True)
        a = ci["node_attrs"][:2]
        op = blocks[i].__dict__[FN._OPS][which]
        ok = torch.autograd.gradcheck(lambda t: op(t, a), (x,), eps=1e-6, atol=1e-7, rtol=1e-6, fast_mode=True)
        RES["misc"][f"gradcheck_L{i}.{which}"] = bool(ok)
except Exception as e:
    RES["misc"]["gradcheck"] = f"FAIL {e!r}"
# double backward must raise
try:
    x = inputs["aspirin"][0]["message"][:3].clone().requires_grad_(True)
    y = blocks[0].__dict__[FN._OPS]["out"](x, inputs["aspirin"][0]["node_attrs"][:3])
    g, = torch.autograd.grad(y.sum(), x, create_graph=True)
    try:
        g.sum().backward()
        RES["misc"]["double_backward"] = "did NOT raise"
    except RuntimeError as e:
        RES["misc"]["double_backward"] = f"raises: {str(e)[:120]}"
except Exception as e:
    RES["misc"]["double_backward"] = f"raises early: {str(e)[:160]}"
print("gradcheck", {k: v for k, v in RES["misc"].items() if "grad" in k or "double" in k}, flush=True)
save()

# ---------------------------------------------------------------------- 3. CUDA graph capture of fused op fwd+bwd + bench
def bench(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3  # us


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


for nm in ["aspirin", "water648"]:
    for i, which in ((0, "out"), (1, "out"), (1, "sc")):
        ci = inputs[nm][i]
        x0 = (ci["message"] if which == "out" else ci["node_feats"]).clone()
        a = ci["node_attrs"].clone()
        op = blocks[i].__dict__[FN._OPS][which]
        xs = x0.clone().requires_grad_(True)
        gy = torch.randn_like(op(x0, a))

        def fused_fb():
            y = op(xs, a)
            gx, = torch.autograd.grad(y, xs, gy)
            return y, gx

        def ref_fb():
            y = ref_ops(i, xs, a)["out"] if which == "out" else ref_sc(i, xs, a)
            gx, = torch.autograd.grad(y, xs, gy)
            return y, gx

        def fused_f():
            with torch.no_grad():
                return op(xs, a)

        def ref_f():
            with torch.no_grad():
                return ref_ops(i, xs, a)["out"] if which == "out" else ref_sc(i, xs, a)
        key = f"{nm}/L{i}.{which}"
        try:
            g, (ys, gxs) = graph_of(fused_fb)
            with torch.no_grad():
                xs.copy_(x0 * 1.37 + 0.01)
            g.replay(); torch.cuda.synchronize()
            ye, gxe = ref_fb()
            RES["graph"][key] = dict(capture="ok", replay_vs_stock_fwd=errs(ys, ye), replay_vs_stock_bwd=errs(gxs, gxe))
            gr_, _ = graph_of(ref_fb)
            gf_f, _ = graph_of(fused_f); gr_f, _ = graph_of(ref_f)
            RES["bench"][key] = dict(
                N=x0.shape[0],
                eager_fwd_us=dict(stock=bench(ref_f), fused=bench(fused_f)),
                eager_fwdbwd_us=dict(stock=bench(ref_fb), fused=bench(fused_fb)),
                graph_fwd_us=dict(stock=bench(gr_f.replay), fused=bench(gf_f.replay)),
                graph_fwdbwd_us=dict(stock=bench(gr_.replay), fused=bench(g.replay)))
            with torch.no_grad():
                xs.copy_(x0)
        except Exception:
            RES["graph"][key] = f"FAIL {traceback.format_exc()}"
        print(key, RES["graph"].get(key), RES["bench"].get(key), flush=True)
# kernel launch counts under profiler (fused fwd+bwd, L0, aspirin)
try:
    from torch.profiler import profile, ProfilerActivity
    ci = inputs["aspirin"][0]; xs = ci["message"].clone().requires_grad_(True); a = ci["node_attrs"]
    for nmk, fn in (("fused", lambda: torch.autograd.grad(blocks[0].__dict__[FN._OPS]["out"](xs, a).sum(), xs)),
                    ("stock", lambda: torch.autograd.grad(ref_ops(0, xs, a)["out"].sum(), xs))):
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        RES["misc"][f"L0_fwdbwd_cuda_kernels_{nmk}(incl sum/ones)"] = sum(1 for e in p.events() if e.device_type.name == "CUDA")
except Exception as e:
    RES["misc"]["profile"] = repr(e)
save()

# ---------------------------------------------------------------------- 4. whole model: e3nn and hybrid backends
MSYS = ["H2O", "aspirin", "water648", "wbox648", "single_C", "far_OO"]
FN.uninstall(model, handle)
assert "forward" not in model.interactions[0].__dict__
for nm in MSYS:
    at = load(nm)
    stress = bool(at.pbc.any())
    bd = batch_dict(calc, at)
    n = len(at)
    try:
        ref = run_model(model, bd, stress)
        ref2 = run_model(model, bd, stress)
        mods0 = [(id(b.linear), id(b.skip_tp), id(b.reshape), id(b.conv_tp)) for b in model.interactions]
        h = FN.install(model)
        fu = run_model(model, bd, stress)
        fu2 = run_model(model, bd, stress)
        FN.uninstall(model, h); FN.uninstall(model, h)
        back = run_model(model, bd, stress)
        mods1 = [(id(b.linear), id(b.skip_tp), id(b.reshape), id(b.conv_tp)) for b in model.interactions]
        RES["model"][f"e3nn/{nm}"] = dict(n=n, fused_vs_stock=cmp_model(fu, ref, n),
            stock_repeat_noise=cmp_model(ref2, ref, n), fused_repeat=cmp_model(fu2, fu, n),
            uninstalled_vs_stock=cmp_model(back, ref, n),
            uninstall_restores_objects=bool(mods0 == mods1 and all("forward" not in b.__dict__ and FN._OPS not in b.__dict__
                                                                   for b in model.interactions)))
    except Exception:
        RES["model"][f"e3nn/{nm}"] = f"FAIL {traceback.format_exc()}"
        FN.uninstall(model)
    print("model e3nn", nm, RES["model"][f"e3nn/{nm}"], flush=True)
save()
refs = {}
for nm in MSYS:
    at = load(nm)
    refs[nm] = run_model(model, batch_dict(calc, at), bool(at.pbc.any()))
del calc, model
torch.cuda.empty_cache()
try:
    hcalc, hinfo = HB.build_calc(conv="flashtp", sc="cueq", dtype=torch.float64, check_indices=False)
    hm = hcalc.models[0]
    for nm in MSYS:
        at = load(nm)
        stress = bool(at.pbc.any())
        bd = batch_dict(hcalc, at)
        n = len(at)
        try:
            hy = run_model(hm, bd, stress)
            hy2 = run_model(hm, bd, stress)
            h = FN.install(hm)
            hf = run_model(hm, bd, stress)
            FN.uninstall(hm, h)
            back = run_model(hm, bd, stress)
            RES["model"][f"hybrid/{nm}"] = dict(n=n, hybrid_vs_e3nn=cmp_model(hy, refs[nm], n),
                                                hybrid_fused_vs_e3nn=cmp_model(hf, refs[nm], n),
                                                hybrid_fused_vs_hybrid=cmp_model(hf, hy, n),
                                                hybrid_repeat_noise=cmp_model(hy2, hy, n),
                                                uninstalled_vs_hybrid=cmp_model(back, hy, n))
        except Exception:
            RES["model"][f"hybrid/{nm}"] = f"FAIL {traceback.format_exc()}"
            FN.uninstall(hm)
        print("model hybrid", nm, RES["model"][f"hybrid/{nm}"], flush=True)
    # double install must raise
    FN.install(hm)
    try:
        FN.install(hm); RES["misc"]["double_install"] = "did NOT raise"
    except FN.FuseNodeError as e:
        RES["misc"]["double_install"] = f"raises: {e}"
    FN.uninstall(hm)
except Exception:
    RES["model"]["hybrid"] = f"FAIL {traceback.format_exc()}"
save()
print("DONE", flush=True)
