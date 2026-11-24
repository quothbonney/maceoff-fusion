"""Capture real conv inputs (x after linear_up, y edge_attrs, w tp_weights incl. cutoff, edge_index) per interaction
from stock e3nn fp64 MACE-OFF23 medium on each system. Saves to $CAPDIR/<sys>_L<i>.pt. Run in venv-cueq."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import common as C

os.makedirs(C.CAPDIR, exist_ok=True)
calc = C.load_calc()
model = calc.models[0]
store = {}
hooks = []
for i, blk in enumerate(model.interactions):
    def pre_blk(mod, args, kwargs, i=i):
        store[(i, "ei")] = kwargs.get("edge_index", args[4] if len(args) > 4 else None).detach().clone()
    def post_up(mod, args, out, i=i):
        store[(i, "x")] = out.detach().clone()
    def pre_tp(mod, args, i=i):
        store[(i, "y")] = args[1].detach().clone(); store[(i, "w")] = args[2].detach().clone()
    hooks += [blk.register_forward_pre_hook(pre_blk, with_kwargs=True), blk.linear_up.register_forward_hook(post_up),
              blk.conv_tp.register_forward_pre_hook(pre_tp)]

manifest = {}
for name in sys.argv[1:] or C.SYSTEMS:
    atoms = C.load_atoms(name)
    atoms.calc = calc
    store.clear()
    calc.calculate(atoms, ["energy", "forces"])
    for i in range(len(model.interactions)):
        d = dict(x=store[(i, "x")], y=store[(i, "y")], w=store[(i, "w")], ei=store[(i, "ei")])
        sorted_src = bool((d["ei"][0][1:] >= d["ei"][0][:-1]).all().item()) if d["ei"].shape[1] > 1 else True
        torch.save({k: v.cpu() for k, v in d.items()}, f"{C.CAPDIR}/{name}_L{i}.pt")
        h = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in d.values())).hexdigest()[:16]
        manifest[f"{name}_L{i}"] = dict(N=d["x"].shape[0], E=d["ei"].shape[1], d1=d["x"].shape[1], W=d["w"].shape[1],
                                        dtype=str(d["x"].dtype), sorted_by_src=sorted_src, sha=h)
        print(name, i, manifest[f"{name}_L{i}"], flush=True)
json.dump(dict(env=C.env_info(), caps=manifest), open(f"{C.R}/results/oeq8/capture_manifest.json", "w"), indent=1)
