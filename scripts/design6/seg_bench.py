"""design6 segment microbenchmark (MACE-OFF23 medium, fp64, hybrid = FlashTP conv + cueq SC).
For each candidate fusion boundary, time the isolated segment fwd + bwd (bwd wrt the tensors that carry grad in the
real force pass) two ways: eager wall (sync-bracketed, includes host/launch overhead) and CUDA-graph replay
(= GPU-only time). Also: whole-model eager fwd_bwd wall, profiler kernel-sum, and a CUDA-graph capture attempt of
FlashTP (expected to fail: legacy default stream + cudaMemcpyToSymbol per call).
Usage: python -I seg_bench.py <out_json> [systems...]"""
import os, sys, json, time, traceback, contextlib
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_v2"); sys.path.insert(0, f"{R}/scripts/workloads")
import hybrid_v2 as HB
import numpy as np, torch
from mace.tools import torch_tools
from systems2 import load_system

out_path = sys.argv[1]; names = sys.argv[2:] or ["aspirin", "org192", "water648"]
M = f"{R}/models/MACE-OFF23_medium.model"
sync = torch.cuda.synchronize
NREP = 40


@contextlib.contextmanager
def quiet():  # FlashTP fwd prints shared_memory_bytes via std::cout every call
    sys.stdout.flush(); saved = os.dup(1); nul = os.open(os.devnull, os.O_WRONLY); os.dup2(nul, 1)
    try: yield
    finally: sys.stdout.flush(); os.dup2(saved, 1); os.close(nul); os.close(saved)


def make_bd(calc, atoms):
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v): b[k] = v.to(dtype=torch.float64)
    return b.to_dict()


def eager_ms(fn, n=NREP):
    for _ in range(5): fn()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def graph_ms(fn, n=NREP):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(3): g.replay()
    sync()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n):
        e0.record(); g.replay(); e1.record(); sync(); ts.append(e0.elapsed_time(e1))
    del g
    return float(np.median(ts))


def kernel_sum_ms(fn, n=5):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3): fn()
    sync()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(n): fn()
        sync()
    tot = 0.0; cnt = 0
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total; cnt += 1
    return tot / 1e3 / n, cnt / n


def seg(fn_build):
    """fn_build() -> callable doing fwd+bwd on fixed tensors."""
    r = {}
    fn = fn_build()
    r["eager_ms"] = eager_ms(fn)
    try:
        r["graph_ms"] = graph_ms(fn)
    except Exception as ex:
        r["graph_ms"] = None; r["graph_err"] = repr(ex)[:300]
    try:
        r["kernel_sum_ms"], r["kernels_per_call"] = kernel_sum_ms(fn)
    except Exception as ex:
        r["kernel_err"] = repr(ex)[:300]
    return r


def grad_fn(f, inputs):
    """returns callable: out = f(*inputs); autograd.grad(out, inputs that require grad, random seed)."""
    with torch.no_grad():
        pass
    out0 = f(*inputs)
    gseed = torch.randn_like(out0)
    req = [t for t in inputs if torch.is_tensor(t) and t.requires_grad]
    def run():
        o = f(*inputs)
        return torch.autograd.grad([o], req, grad_outputs=[gseed])
    return run


res = {"meta": dict(job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(), note="hybrid_v2 conv=flashtp sc=cueq check_indices=False"),
       "systems": {}}
with quiet():
    calc, info = HB.build_calc(M, conv="flashtp", sc="cueq", check_indices=False)
model = calc.models[0]
res["meta"]["backend_map"] = str(info.get("backend_map"))

cap = {}
def pre_hook(name):
    def h(mod, args, kwargs):
        cap[name] = (tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args),
                     {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()})
    return h
hooks = []
for i in range(2):
    it = model.interactions[i]
    for nm, m in [("inter", it), ("mlp", it.conv_tp_weights), ("conv", it.conv_tp), ("linear", it.linear),
                  ("skip", it.skip_tp)]:
        hooks.append(m.register_forward_pre_hook(pre_hook(f"{nm}{i}"), with_kwargs=True))
    hooks.append(model.products[i].register_forward_pre_hook(pre_hook(f"prod{i}"), with_kwargs=True))
    hooks.append(model.readouts[i].register_forward_pre_hook(pre_hook(f"read{i}"), with_kwargs=True))

for name in names:
    atoms = load_system(name)
    bd = make_bd(calc, atoms)
    E = int(bd["edge_index"].shape[1]); N = int(bd["positions"].shape[0])
    r = {"n_atoms": N, "n_edges": E}
    print(f"=== {name} N={N} E={E}", flush=True)

    def full():
        with torch_tools.default_dtype("float64"):
            return model(bd, compute_force=True, training=False)
    with quiet():
        full(); sync()
        r["model_fwd_bwd_eager_ms"] = eager_ms(full, 30)
        r["model_kernel_sum_ms"], r["model_kernels_per_call"] = kernel_sum_ms(full)
    for h in hooks: pass
    sync()
    segs = {}
    rq = lambda t: t.detach().clone().requires_grad_(True)
    for i in range(2):
        it = model.interactions[i]
        a, kw = cap[f"inter{i}"]
        cutoff = rq(kw["cutoff"]) if kw.get("cutoff") is not None else None; ef = rq(kw["edge_feats"])
        r[f"L{i}_cutoff_is_none"] = cutoff is None
        mlp = it.conv_tp_weights
        # S1: radial MLP + cutoff multiply (fwd + bwd wrt edge_feats and cutoff)
        if cutoff is None:
            segs[f"L{i}_radial_mlp_cutoff"] = seg(lambda: grad_fn(lambda x: mlp(x), [ef]))
        else:
            segs[f"L{i}_radial_mlp_cutoff"] = seg(lambda: grad_fn(lambda x, c: mlp(x) * c, [ef, cutoff]))
        # S1b: only the materialising tail (last 64->W layer + cutoff mul), emulated with same shapes
        Wn = mlp(ef).shape[1]
        h3 = torch.randn(E, 64, dtype=torch.float64, device="cuda", requires_grad=True)
        Wm = torch.randn(64, Wn, dtype=torch.float64, device="cuda") / 8.0
        segs[f"L{i}_mlp_tail_64xW"] = seg(lambda: grad_fn(lambda x: x @ Wm, [h3]))
        # S2: linear (+ /avg) (+ skip_tp for layer 0) on the scattered message
        ca, ckw = cap[f"linear{i}"]
        msg = rq(ca[0]); node_attrs = kw["node_attrs"]
        if i == 0:
            f2 = lambda m: it.skip_tp(it.linear(m) / it.avg_num_neighbors, node_attrs)
        else:
            f2 = lambda m: it.linear(m) / it.avg_num_neighbors
        segs[f"L{i}_linear_avg_skip"] = seg(lambda: grad_fn(f2, [msg]))
        # S3: product block (cueq SC + linear + sc) and readout
        pa, pkw = cap[f"prod{i}"]
        nf = rq(pkw["node_feats"]); sc = pkw["sc"]; sc = rq(sc) if sc is not None else None
        prod = model.products[i]; na = pkw["node_attrs"]
        if sc is not None:
            segs[f"L{i}_product_sc_linear"] = seg(lambda: grad_fn(lambda x, s: prod(node_feats=x, sc=s, node_attrs=na), [nf, sc]))
        else:
            segs[f"L{i}_product_sc_linear"] = seg(lambda: grad_fn(lambda x: prod(node_feats=x, sc=None, node_attrs=na), [nf]))
        ra, rkw = cap[f"read{i}"]
        rf = rq(ra[0]); rh = ra[1]; ro = model.readouts[i]
        segs[f"L{i}_readout"] = seg(lambda: grad_fn(lambda x: ro(x, rh), [rf]))
        # FlashTP conv (eager only; graph capture attempted at the very end)
        cva, _ = cap[f"conv{i}"]
        x1 = rq(cva[0]) if i > 0 else cva[0]
        sh = rq(cva[1]); tw = rq(cva[2]); ei = cva[3]
        with quiet():
            fc = grad_fn(lambda a1, a2, a3: it.conv_tp(a1, a2, a3, ei), [x1, sh, tw])
            segs[f"L{i}_flashtp_conv"] = {"eager_ms": eager_ms(fc)}
            try:
                segs[f"L{i}_flashtp_conv"]["kernel_sum_ms"], segs[f"L{i}_flashtp_conv"]["kernels_per_call"] = kernel_sum_ms(fc)
            except Exception as ex:
                segs[f"L{i}_flashtp_conv"]["kernel_err"] = repr(ex)[:200]
    # embedding: sh + radial basis (fwd + bwd wrt vectors)
    vec = (bd["positions"][bd["edge_index"][1]] - bd["positions"][bd["edge_index"][0]]).detach().requires_grad_(True)
    na0 = bd["node_attrs"]; eidx = bd["edge_index"]
    def emb(v):
        L = torch.linalg.norm(v, dim=-1, keepdim=True)
        ea = model.spherical_harmonics(v)
        ef, cut = model.radial_embedding(L, na0, eidx, model.atomic_numbers)
        return torch.cat([ea, ef] + ([] if cut is None else [cut.reshape(ef.shape[0], -1)]), dim=1)
    segs["embed_sh_radial"] = seg(lambda: grad_fn(emb, [vec]))
    r["segments"] = segs
    res["systems"][name] = r
    print(json.dumps(r, indent=1), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)

# FlashTP capture attempt (last; may poison the context)
try:
    it = model.interactions[0]; cva, _ = cap["conv0"]
    sh = cva[1].detach().clone().requires_grad_(True); tw = cva[2].detach().clone().requires_grad_(True)
    with quiet():
        fc = grad_fn(lambda a2, a3: it.conv_tp(cva[0], a2, a3, cva[3]), [sh, tw])
        res["flashtp_capture"] = {"graph_ms": graph_ms(fc)}
except Exception as ex:
    res["flashtp_capture"] = {"error": repr(ex)[:500]}
print("flashtp_capture", res["flashtp_capture"], flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("DONE", flush=True)
