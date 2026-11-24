"""validate6 check 4: eager FlashTP graphcap vs cgpatch on REAL MACE layer inputs (per-op out + 1st + 2nd order grads)
and full-model eager E/F.  usage: python -I v_parity.py run graphcap|cgpatch DIR ;  python -I v_parity.py compare DIR OUT.json"""
import os, sys, json
R = os.environ["R"]
mode = sys.argv[1]
if mode == "compare":
    import torch
    D = sys.argv[2]
    a, b = torch.load(f"{D}/out_graphcap.pt"), torch.load(f"{D}/out_cgpatch.pt")
    rows = {}
    for k in a:
        if k == "meta":
            continue
        x, y = a[k], b[k]
        d = (x - y).abs()
        rows[k] = dict(shape=list(x.shape), max_abs=float(d.max()), max_ref=float(y.abs().max()),
                       bitwise_equal=bool(torch.equal(x, y)), rel=float(d.max() / max(float(y.abs().max()), 1e-300)))
    out = dict(meta_graphcap=a["meta"], meta_cgpatch=b["meta"], rows=rows,
               max_rel=max(r["rel"] for r in rows.values()), n_bitwise_equal=sum(r["bitwise_equal"] for r in rows.values()),
               n=len(rows))
    out["pass_"] = out["max_rel"] < 1e-12
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    for k, r in rows.items():
        print(k, r)
    print("max_rel", out["max_rel"], "bitwise", out["n_bitwise_equal"], "/", out["n"])
    sys.exit(0)

build, D = sys.argv[2], sys.argv[3]
if build == "graphcap":
    sys.path.insert(0, f"{R}/scripts/cudagraph6"); import hybrid_gc as HB
else:
    sys.path.insert(0, f"{R}/scripts/flash_v2"); import hybrid_v2 as HB
assert HB.flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-{build}/"), HB.flashTP_e3nn.__file__
sys.path.insert(1, f"{R}/scripts/graph")
import torch, ase.io
from mace.tools import torch_tools
from proto_graph import ProtoBuilder
calc, _ = HB.build_calc(conv="flashtp", sc="e3nn")
model = calc.models[0]; pb = ProtoBuilder(calc)
inp_path = f"{D}/layer_inputs.pt"
outs = dict(meta=dict(flashTP=HB.flashTP_e3nn.__file__, job=os.environ.get("SLURM_JOB_ID")))
systems = ["aspirin", "water192", "wbox648"]
if not os.path.exists(inp_path):
    cap = {}
    def mk(tag):
        def hook(mod, args):
            cap[tag] = [t.detach().clone() for t in args]
        return hook
    for s in systems:
        at = ase.io.read(f"{R}/results/workloads/systems/{s}.extxyz"); bd = pb.build(at)
        hs = [b.conv_tp.register_forward_pre_hook(mk(f"{s}/L{i}")) for i, b in enumerate(model.interactions)]
        with torch_tools.default_dtype("float64"):
            model(bd, compute_force=True, training=False)
        for h in hs:
            h.remove()
    torch.save(cap, inp_path)
inputs = torch.load(inp_path)
for s in systems:  # full model eager
    at = ase.io.read(f"{R}/results/workloads/systems/{s}.extxyz"); bd = pb.build(at)
    with torch_tools.default_dtype("float64"):
        o = model(bd, compute_force=True, training=False)
    outs[f"model/{s}/E"] = o["energy"].detach().cpu(); outs[f"model/{s}/F"] = o["forces"].detach().cpu()
for tag, (nf, ea, w, ei) in inputs.items():
    li = int(tag.split("/L")[1]); ftp = model.interactions[li].conv_tp.ftp
    g = torch.Generator().manual_seed(li * 7 + len(tag))
    nf, ea, w = [t.clone().requires_grad_(True) for t in (nf, ea, w)]
    src, dst = ei[0].to(torch.int32).contiguous(), ei[1].to(torch.int32).contiguous()
    y = ftp(nf, ea, w, src, dst)
    ct = torch.randn(y.shape, generator=g, dtype=torch.float64).to(y.device)
    gn, ge, gw = torch.autograd.grad((y * ct).sum(), (nf, ea, w), create_graph=True)
    c2 = [torch.randn(t.shape, generator=g, dtype=torch.float64).to(t.device) for t in (gn, ge, gw)]
    s2 = (gn * c2[0]).sum() + (ge * c2[1]).sum() + (gw * c2[2]).sum()
    dd = torch.autograd.grad(s2, (nf, ea, w), allow_unused=True)
    outs[f"{tag}/out"] = y.detach().cpu()
    for nm, t in zip(("d_in1", "d_in2", "d_w"), (gn, ge, gw)):
        outs[f"{tag}/{nm}"] = t.detach().cpu()
    for nm, t in zip(("dd_in1", "dd_in2", "dd_w"), dd):
        if t is not None:
            outs[f"{tag}/{nm}"] = t.detach().cpu()
torch.save(outs, f"{D}/out_{build}.pt")
print("saved", build, len(outs))
