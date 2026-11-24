"""water648 breakdown: (1) isolated conv-TP+scatter timing per interaction layer with the real captured inputs
(e3nn conv_tp + scatter_sum vs FlashConv), fwd and fwd+bwd, compared with whole-model fwd / fwd+bwd;
(2) torch.profiler CUDA kernel table for one model(compute_force=True) per backend.
Usage: python -I prof.py <out_json> <log_prefix>"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flash_proto"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np, torch
from mace.calculators import MACECalculator
from mace.tools import torch_tools
from mace.tools.scatter import scatter_sum
import flashtp_adapter as FA
from systems import get_system

R = os.environ["R"]; out_path, logp = sys.argv[1], sys.argv[2]
sync = torch.cuda.synchronize
calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64", compute_stress=False)
model = calc.models[0]
atoms = get_system("water648")
b = calc._clone_batch(calc._atoms_to_batch(atoms)); bd = b.to_dict()

# capture conv_tp inputs (e3nn path) per layer
cap = {}
def mk_hook(li):
    def hook(mod, args):
        x, ea, w = args
        cap[li] = dict(x_rg=x.requires_grad, ea_rg=ea.requires_grad, w_rg=w.requires_grad, ea=ea.detach().clone(), w=w.detach().clone())
    return hook
hs = [model.interactions[li].conv_tp.register_forward_pre_hook(mk_hook(li)) for li in range(len(model.interactions))]
def lu_hook(li):
    def hook(mod, args, out): cap.setdefault(f"x{li}", out.detach().clone())
    return hook
hs += [model.interactions[li].linear_up.register_forward_hook(lu_hook(li)) for li in range(len(model.interactions))]
with torch_tools.default_dtype("float64"): model(bd, compute_force=True)
for h in hs: h.remove()
ei = bd["edge_index"]; N = bd["positions"].shape[0]
saved = FA.install(model); flash = {li: model.interactions[li].conv_tp for li, _ in saved}; FA.uninstall(model, saved)

def bench(fn, n=50, w=10):
    for _ in range(w): fn()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return dict(median=float(np.median(ts)), p10=float(np.percentile(ts, 10)), p90=float(np.percentile(ts, 90)))

res = {"n_atoms": N, "n_edges": int(ei.shape[1]), "layers": {}}
for li, orig in saved:
    c = cap[li]; x0 = cap[f"x{li}"]
    def inputs():
        x = x0.clone().requires_grad_(c["x_rg"]); ea = c["ea"].clone().requires_grad_(c["ea_rg"]); w = c["w"].clone().requires_grad_(c["w_rg"])
        return x, ea, w
    def e3(bw):
        x, ea, w = inputs()
        m = scatter_sum(src=orig(x[ei[0]], ea, w), index=ei[1], dim=0, dim_size=N)
        if bw: torch.autograd.grad(m, [t for t in (x, ea, w) if t.requires_grad], grad_outputs=torch.ones_like(m))
    def fl(bw):
        x, ea, w = inputs()
        m = flash[li](x, ea, w, ei)
        if bw: torch.autograd.grad(m, [t for t in (x, ea, w) if t.requires_grad], grad_outputs=torch.ones_like(m))
    sys.stdout.flush()
    res["layers"][li] = dict(requires_grad=dict(x=c["x_rg"], edge_attrs=c["ea_rg"], tp_weights=c["w_rg"]),
                             shapes=dict(x=list(x0.shape), edge_attrs=list(c["ea"].shape), tp_weights=list(c["w"].shape)),
                             e3nn_fwd=bench(lambda: e3(False)), e3nn_fwdbwd=bench(lambda: e3(True)),
                             flash_fwd=bench(lambda: fl(False)), flash_fwdbwd=bench(lambda: fl(True)))
    print("LAYER", li, json.dumps(res["layers"][li]), flush=True)

def set_backend(name):
    for li, orig in saved:
        blk = model.interactions[li]
        if name == "e3nn":
            blk.conv_tp = orig
            if hasattr(blk, "conv_fusion"): del blk.conv_fusion
        else: blk.conv_tp = flash[li]; blk.conv_fusion = True

from torch.profiler import profile, ProfilerActivity
for be in ["e3nn", "flash"]:
    set_backend(be)
    def fwd():
        with torch_tools.default_dtype("float64"): model(bd, compute_force=False)
    def fb():
        with torch_tools.default_dtype("float64"): model(bd, compute_force=True)
    res[f"model_{be}_fwd"] = bench(fwd); res[f"model_{be}_fwdbwd"] = bench(fb)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(5): fb()
        sync()
    ka = p.key_averages()
    tbl = ka.table(sort_by="self_cuda_time_total" if hasattr(ka[0], "self_cuda_time_total") else "self_device_time_total", row_limit=25)
    open(f"{logp}_profile_{be}.txt", "w").write(tbl)
    tot = 0.0; kern = []
    for e in ka:
        st = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if e.device_type.name == "CUDA" if hasattr(e, "device_type") else False:
            tot += st; kern.append((e.key, st / 5 / 1e3, e.count // 5))
    kern.sort(key=lambda t: -t[1])
    res[f"profile_{be}"] = dict(total_gpu_kernel_ms_per_call=tot / 5 / 1e3,
                                top_kernels_ms_per_call=[dict(name=k[:120], ms=m, calls=c) for k, m, c in kern[:15]])
    print("PROF", be, json.dumps(res[f"profile_{be}"])[:3000], flush=True)
set_backend("e3nn")
for k in ("fwd", "fwdbwd"):
    res[f"conv_sum_e3nn_{k}_ms"] = sum(res["layers"][li][f"e3nn_{k}"]["median"] for li in res["layers"])
    res[f"conv_sum_flash_{k}_ms"] = sum(res["layers"][li][f"flash_{k}"]["median"] for li in res["layers"])
    res[f"conv_fraction_of_model_e3nn_{k}"] = res[f"conv_sum_e3nn_{k}_ms"] / res[f"model_e3nn_{k}"]["median"]
print("SUMMARY", json.dumps({k: v for k, v in res.items() if k.startswith("conv_") or k.startswith("model_")}), flush=True)
json.dump(res, open(out_path, "w"), indent=1, default=str)
