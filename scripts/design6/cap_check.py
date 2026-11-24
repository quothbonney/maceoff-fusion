"""Does torch.cuda.graph capture of the shipped FlashTP (cgpatch) conv contain the FlashTP kernels? Replay vs eager."""
import os, sys, json, torch
R = os.environ["R"]; sys.path.insert(0, f"{R}/scripts/flash_v2"); sys.path.insert(0, f"{R}/scripts/workloads")
import hybrid_v2 as HB
from systems2 import load_system
calc, info = HB.build_calc(f"{R}/models/MACE-OFF23_medium.model", conv="flashtp", sc="cueq", check_indices=False)
model = calc.models[0]; cap = {}
model.interactions[0].conv_tp.register_forward_pre_hook(lambda m, a: cap.__setitem__("c", [t.detach().clone() if torch.is_tensor(t) else t for t in a]))
from mace.tools import torch_tools
at = load_system("water648"); b = calc._clone_batch(calc._atoms_to_batch(at))
bd = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.to_dict().items()}
with torch_tools.default_dtype("float64"): model(bd, compute_force=True)
x, sh, tw, ei = cap["c"]; conv = model.interactions[0].conv_tp
sh = sh.requires_grad_(True); tw = tw.requires_grad_(True)
def run():
    o = conv(x, sh, tw, ei); g = torch.randn_like(o) if not hasattr(run, "g") else run.g; run.g = g
    return (o,) + torch.autograd.grad([o], [sh, tw], grad_outputs=[g])
ref = [t.clone() for t in run()]
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): run()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): outs = run()
for t in outs: t.zero_()
g.replay(); torch.cuda.synchronize()
r = {n: float((a - b_).abs().max()) for n, a, b_ in zip(["out", "dsh", "dtw"], outs, ref)}
r["ref_absmax"] = {n: float(a.abs().max()) for n, a in zip(["out", "dsh", "dtw"], ref)}
print("CAPCHECK", json.dumps(r), flush=True)
json.dump(r, open(f"{R}/results/design6/flashtp_capture_check.json", "w"), indent=1)
