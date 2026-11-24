"""Per-layer parity on real inputs: FlashConv(fp64CG) vs e3nn conv_tp+scatter_sum; CueqSC vs e3nn SymmetricContraction.
Inputs are captured (forward pre-hooks) from a hybrid-model forward on the real system. Usage: layer_parity.py out.json"""
import json, sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from mace.tools.scatter import scatter_sum
out_path = sys.argv[1]
calc_r, info_r = HB.build_calc(MODEL, conv="e3nn", sc="e3nn")
calc_h, info_h = HB.build_calc(MODEL)
mr, mh = calc_r.models[0], calc_h.models[0]
res = dict(env=env_info(), hybrid_info=info_h, ref_info=info_r, systems={})
print(json.dumps(info_h, indent=1), flush=True)
for name in ["H2O", "water648"]:
    cap = {}
    hooks = [b.conv_tp.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(f"conv{i}", [x.detach().clone() for x in a]))
             for i, b in enumerate(mh.interactions)]
    hooks += [p.symmetric_contractions.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(f"sc{i}", [x.detach().clone() for x in a]))
              for i, p in enumerate(mh.products)]
    at = get_system(name); at.calc = calc_h; at.get_forces()
    for h in hooks: h.remove()
    sysres = {}
    for i in range(len(mh.interactions)):
        nf, ea, w, ei = cap[f"conv{i}"]
        ins = [x.clone().requires_grad_(True) for x in (nf, ea, w)]
        ot = mh.interactions[i].conv_tp(*ins, ei)
        insr = [x.clone().requires_grad_(True) for x in (nf, ea, w)]
        orr = scatter_sum(mr.interactions[i].conv_tp(insr[0][ei[0]], insr[1], insr[2]), ei[1], dim=0, dim_size=nf.shape[0])
        g = torch.randn(orr.shape, generator=torch.Generator(device="cuda").manual_seed(1234 + i), device="cuda", dtype=orr.dtype)
        gt = torch.autograd.grad(ot, ins, g); gr = torch.autograd.grad(orr, insr, g)
        sysres[f"interactions.{i}.conv"] = dict(n_nodes=nf.shape[0], n_edges=ei.shape[1], out=metr(ot, orr),
                                                d_node_feats=metr(gt[0], gr[0]), d_edge_attrs=metr(gt[1], gr[1]), d_tp_weights=metr(gt[2], gr[2]))
    for i in range(len(mh.products)):
        nf, na = cap[f"sc{i}"]
        xt = nf.clone().requires_grad_(True); xr = nf.clone().requires_grad_(True)
        ot = mh.products[i].symmetric_contractions(xt, na); orr = mr.products[i].symmetric_contractions(xr, na)
        g = torch.randn(orr.shape, generator=torch.Generator(device="cuda").manual_seed(4321 + i), device="cuda", dtype=orr.dtype)
        gt, = torch.autograd.grad(ot, xt, g); gr, = torch.autograd.grad(orr, xr, g)
        sysres[f"products.{i}.symmetric_contractions"] = dict(in_shape=list(nf.shape), out=metr(ot, orr), d_node_feats=metr(gt, gr))
    res["systems"][name] = sysres
    print(name, json.dumps(sysres, indent=1), flush=True)
json.dump(res, open(out_path, "w"), indent=2)
