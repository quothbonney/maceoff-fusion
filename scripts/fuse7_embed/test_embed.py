"""Local tests for fuse7_embed (GPU). Usage: python -I test_embed.py <out_json>"""
import os, sys, json, time, math
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/fuse7_embed"); sys.path.insert(0, f"{R}/scripts/workloads")
import numpy as np, torch
import fused_embed as FE
from mace.calculators import MACECalculator
from mace.tools import torch_tools
from systems2 import load_system
from ase.build import molecule
from ase import Atoms

torch.set_default_dtype(torch.float64)
out_path = sys.argv[1]
M = f"{R}/models/MACE-OFF23_medium.model"
dev = torch.device("cuda")
sync = torch.cuda.synchronize
res = {"meta": dict(job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name()), "op": {}, "model": {}, "bench": {}, "graph": {}}
fails = []

t0 = time.time(); FE.ext(); res["meta"]["build_s"] = time.time() - t0
calc = MACECalculator(model_paths=M, device="cuda", default_dtype="float64")
model = calc.models[0]
for p in model.parameters(): p.requires_grad_(False)
sh_ref, re_ref = model.spherical_harmonics, model.radial_embedding
res["meta"]["sh"] = repr(sh_ref); res["meta"]["radial"] = repr(re_ref)
res["meta"]["radial_cfg"] = dict(w=re_ref.bessel_fn.bessel_weights.tolist(), pref=float(re_ref.bessel_fn.prefactor),
                                 rmax=float(re_ref.cutoff_fn.r_max), p=int(re_ref.cutoff_fn.p),
                                 has_dt=hasattr(re_ref, "distance_transform"),
                                 apply_cutoff=getattr(re_ref, "apply_cutoff", None))
print(json.dumps(res["meta"], indent=1), flush=True)
sh_f = FE.FusedSphericalHarmonics(sh_ref).to(dev)
re_f = FE.FusedRadialEmbedding(re_ref).to(dev)


def make_bd(atoms):
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    return b.to_dict()


def edge_vectors(atoms):
    bd = make_bd(atoms)
    pos = bd["positions"]; ei = bd["edge_index"]; sh = bd["shifts"]
    return (pos[ei[1]] - pos[ei[0]] + sh).detach()


def ref_fn(v):
    lengths = torch.linalg.norm(v, dim=-1, keepdim=True)
    a = sh_ref(v); f, c = re_ref(lengths, None, None, None)
    assert c is None
    return a, f


def fused_fn(v):
    lengths = torch.linalg.norm(v, dim=-1, keepdim=True)
    a = sh_f(v); f, c = re_f(lengths)
    return a, f


def compare_op(name, v):
    v1 = v.clone().requires_grad_(True); v2 = v.clone().requires_grad_(True)
    a1, f1 = ref_fn(v1); a2, f2 = fused_fn(v2)
    E = v.shape[0]
    g = torch.Generator(device="cuda").manual_seed(0)
    ga = torch.randn(a1.shape, generator=g, device="cuda"); gf = torch.randn(f1.shape, generator=g, device="cuda")
    r = dict(E=E)
    r["sh_max_abs"] = float((a1 - a2).abs().max()) if E else 0.0
    r["rad_max_abs"] = float((f1 - f2).abs().max()) if E else 0.0
    if E:
        (gv1,) = torch.autograd.grad([a1, f1], [v1], [ga, gf])
        (gv2,) = torch.autograd.grad([a2, f2], [v2], [ga, gf])
        r["grad_max_abs"] = float((gv1 - gv2).abs().max())
        r["grad_max_rel"] = float(((gv1 - gv2).abs() / gv1.abs().clamp_min(1e-30)).max())
        r["grad_rel_norm"] = float((gv1 - gv2).norm() / gv1.norm())
        # separate sh-only / radial-only grads
        v3 = v.clone().requires_grad_(True); v4 = v.clone().requires_grad_(True)
        (gs1,) = torch.autograd.grad(ref_fn(v3)[0], v3, ga); (gs2,) = torch.autograd.grad(fused_fn(v4)[0], v4, ga)
        r["grad_sh_max_abs"] = float((gs1 - gs2).abs().max())
        v3 = v.clone().requires_grad_(True); v4 = v.clone().requires_grad_(True)
        (gr1,) = torch.autograd.grad(ref_fn(v3)[1], v3, gf); (gr2,) = torch.autograd.grad(fused_fn(v4)[1], v4, gf)
        r["grad_rad_max_abs"] = float((gr1 - gr2).abs().max())
        r["grad_rad_max_rel_norm"] = float((gr1 - gr2).norm() / gr1.norm().clamp_min(1e-300))
    else:
        (gv2,) = torch.autograd.grad([a2.sum() + f2.sum()], [v2], allow_unused=True)
        r["shapes"] = [list(a2.shape), list(f2.shape), None if gv2 is None else list(gv2.shape)]
    ok = r["sh_max_abs"] <= 1e-13 and r["rad_max_abs"] <= 1e-13 and r.get("grad_rel_norm", 0.0) <= 1e-12
    r["ok"] = bool(ok)
    if not ok: fails.append(f"op {name}")
    res["op"][name] = r
    print("OP", name, json.dumps(r), flush=True)


systems = {"H2O": molecule("H2O"), "aspirin": load_system("aspirin"), "water648": load_system("water648"),
           "wbox648": load_system("wbox648")}
for nm, at in systems.items():
    compare_op(nm, edge_vectors(at))
gen = torch.Generator(device="cuda").manual_seed(1)
dirs = torch.nn.functional.normalize(torch.randn(20000, 3, generator=gen, device="cuda"), dim=-1)
compare_op("random_0.5_5", dirs * (0.5 + 4.5 * torch.rand(20000, 1, generator=gen, device="cuda")))
compare_op("near_rmax", dirs[:2000] * (5.0 - torch.logspace(-14, -1, 2000, device="cuda")[:, None]))
compare_op("at_and_beyond_rmax", dirs[:4] * torch.tensor([[5.0], [5.0], [5.0 + 1e-12], [5.5]], device="cuda"))
compare_op("axes", torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-2., 0, 0], [0, -3., 0], [0, 0, -4.]],
                                device="cuda"))
compare_op("E0", torch.zeros(0, 3, device="cuda"))

# gradcheck (fp64) on a small set, fused op alone
vs = (dirs[:12] * (0.8 + 4.0 * torch.rand(12, 1, generator=gen, device="cuda"))).clone().requires_grad_(True)
gc = {}
gc["sh"] = bool(torch.autograd.gradcheck(lambda v: sh_f(v), (vs,), eps=1e-6, atol=1e-8, rtol=1e-7))
gc["radial"] = bool(torch.autograd.gradcheck(lambda v: re_f(torch.linalg.norm(v, dim=-1, keepdim=True))[0], (vs,),
                                              eps=1e-6, atol=1e-8, rtol=1e-7))
try:
    vv = vs.detach().clone().requires_grad_(True)
    (gg,) = torch.autograd.grad(sh_f(vv).sum(), vv, create_graph=True)
    gg.sum().backward()
    gc["double_backward"] = "did not raise (unexpected)"; fails.append("double backward did not raise")
except RuntimeError as ex:
    gc["double_backward"] = "raises: " + str(ex)[:120]
res["op"]["gradcheck"] = gc
if not (gc["sh"] and gc["radial"]): fails.append("gradcheck")
print("GRADCHECK", gc, flush=True)


# ---- microbenchmarks: fwd, fwd+bwd; eager (sync bracketed) and graph replay
def eager_ms(fn, n=50):
    for _ in range(5): fn()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def graph_ms(fn, n=50):
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
    return float(np.median(ts))


def kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); sync()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); sync()
    ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(ev), sum(e.device_time_total for e in ev) / 1e3


for nm in ["aspirin", "water648", "wbox648"]:
    v = edge_vectors(systems[nm])
    ga = torch.randn(v.shape[0], 16, device="cuda"); gf = torch.randn(v.shape[0], 8, device="cuda")
    b = {"E": int(v.shape[0])}
    for tag, f in [("eager_e3nn", ref_fn), ("fused", fused_fn)]:
        vr = v.clone().requires_grad_(True)
        fwd = lambda: f(vr)
        def fb():
            a, ff = f(vr)
            return torch.autograd.grad([a, ff], [vr], [ga, gf])
        with torch.no_grad():
            nofwd = lambda: f(v)
            b[f"{tag}_fwd_eager_ms"] = eager_ms(nofwd)
            b[f"{tag}_fwd_graph_ms"] = graph_ms(nofwd)
            b[f"{tag}_fwd_kernels"], b[f"{tag}_fwd_kernel_ms"] = kernels(nofwd)
        b[f"{tag}_fwdbwd_eager_ms"] = eager_ms(fb)
        b[f"{tag}_fwdbwd_graph_ms"] = graph_ms(fb)
        b[f"{tag}_fwdbwd_kernels"], b[f"{tag}_fwdbwd_kernel_ms"] = kernels(fb)
    res["bench"][nm] = b
    print("BENCH", nm, json.dumps(b), flush=True)

# ---- standalone CUDA-graph capture correctness: capture fwd+bwd, replay with new data in static input
for nm in ["water648"]:
    v = edge_vectors(systems[nm])
    vs_ = v.clone().requires_grad_(True)
    ga = torch.randn(v.shape[0], 16, device="cuda"); gf = torch.randn(v.shape[0], 8, device="cuda")
    st = {}
    def body():
        a, ff = fused_fn(vs_)
        (gv,) = torch.autograd.grad([a, ff], [vs_], [ga, gf])
        st["a"], st["f"], st["g"] = a, ff, gv
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s); sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body()
    errs = []
    for it in range(3):
        newv = v * (1.0 + 0.01 * it) + 0.003 * it
        with torch.no_grad(): vs_.copy_(newv)
        g.replay(); sync()
        vr = newv.clone().requires_grad_(True); a, ff = ref_fn(vr)
        (gr,) = torch.autograd.grad([a, ff], [vr], [ga, gf])
        errs.append(max(float((a - st["a"]).abs().max()), float((ff - st["f"]).abs().max()),
                        float((gr - st["g"]).abs().max())))
    res["graph"][nm] = dict(replay_max_abs_err=errs, ok=bool(max(errs) < 1e-12))
    if max(errs) >= 1e-12: fails.append("graph replay")
    print("GRAPH", res["graph"], flush=True)
    del g


# ---- whole model: stock e3nn vs fused-installed
def run_model(atoms, stress):
    bd = make_bd(atoms)
    with torch_tools.default_dtype("float64"):
        o = model(bd, compute_force=True, compute_stress=stress, training=False)
    return (float(o["energy"].sum()), o["forces"].detach().clone(),
            None if not stress or o.get("stress") is None else o["stress"].detach().clone())


msys = {"H2O": (molecule("H2O"), False), "aspirin": (systems["aspirin"], False),
        "water648": (systems["water648"], False), "wbox648": (systems["wbox648"], True),
        "single_C": (Atoms("C", positions=[[0, 0, 0]]), False),
        "O2_20A": (Atoms("OO", positions=[[0, 0, 0], [20, 0, 0]]), False)}
ref = {k: run_model(a, s) for k, (a, s) in msys.items()}
h = FE.install(model)
assert isinstance(model.spherical_harmonics, FE.FusedSphericalHarmonics)
fus = {k: run_model(a, s) for k, (a, s) in msys.items()}
# repeated call + perturbed coordinates
at2 = systems["water648"].copy(); at2.positions += np.random.default_rng(3).normal(scale=0.05, size=at2.positions.shape)
fus["water648_perturbed"] = run_model(at2, False)
FE.uninstall(model, h); FE.uninstall(model, h)
assert model.spherical_harmonics is sh_ref and model.radial_embedding is re_ref
ref["water648_perturbed"] = run_model(at2, False)
after = run_model(systems["aspirin"], False)
res["model"]["uninstall_restores_bitwise"] = bool(after[0] == ref["aspirin"][0] and torch.equal(after[1], ref["aspirin"][1]))
msys["water648_perturbed"] = (at2, False)
for k in ref:
    n = len(msys[k][0])
    e1, f1, s1 = ref[k]; e2, f2, s2 = fus[k]
    r = dict(N=n, E_ref=e1, dE_per_atom=abs(e1 - e2) / n, max_dF=float((f1 - f2).abs().max()),
             rms_dF=float(((f1 - f2) ** 2).mean().sqrt()))
    if s1 is not None: r["max_dstress"] = float((s1 - s2).abs().max())
    r["ok"] = bool(r["dE_per_atom"] <= 1e-11 and r["max_dF"] <= 1e-11 and r.get("max_dstress", 0) <= 1e-11)
    if not r["ok"]: fails.append(f"model {k}")
    res["model"][k] = r
    print("MODEL", k, json.dumps(r), flush=True)

res["describe"] = FE.describe()
res["fails"] = fails; res["all_ok"] = not fails
json.dump(res, open(out_path, "w"), indent=1)
print("ALL_OK" if not fails else f"FAILS {fails}", flush=True)
