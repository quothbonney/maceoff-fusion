"""Follow-up GPU checks. usage: python -I mini_checks.py <mode> <tree> <out>
mode=meta: dump fp32 module metadata (fiber arrays, CG table, ...) for cross-tree bitwise comparison
mode=empty: FlashConv/uvu_TP with 0 edges (CUDA_LAUNCH_BLOCKING=1) ; mode=refrep: e3nn MACE repeatability"""
import os, sys, json
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
mode, tree, out = sys.argv[1:4]
sys.path.insert(0, tree)
import torch, flashTP_e3nn, e3nn.o3 as o3
assert flashTP_e3nn.__file__.startswith(tree)
R = os.environ["R"]
def load(fn, li):
    lines = [l for l in open(f"{R}/src/flashTP/ir_config/{fn}").read().split("\n") if l.strip()]
    tp = json.loads(lines[li])["tp"]; return o3.Irreps(tp[0]), o3.Irreps(tp[1]), o3.Irreps(tp[2]), [tuple(x) for x in tp[3]]
if mode == "meta":
    d = {}
    for fn, li in [("mace/medium.json", 0), ("mace/medium.json", 1), ("sevennet/sevennet-l3i5.json", 1), ("nequip/4_32_5_p_sc.txt", 1)]:
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            m = flashTP_e3nn.uvu_TP(*load(fn, li), device="cuda", dtype=torch.float32)
            torch.set_default_dtype(torch.float32)
            d[f"{fn}:{li}:{default}"] = [t.cpu() if torch.is_tensor(t) else t for t in m.metadata_list] + [getattr(m, "uppath_cnt", None), m.per_block_batch, m.unique_cg_val]
    torch.save(d, out)
elif mode == "empty":
    i1, i2, io, inst = load("mace/medium.json", 0)
    m = flashTP_e3nn.uvu_TP(i1, i2, io, inst, device="cuda", dtype=torch.float64)
    n = 3; w_numel = sum(i.path_shape[0] if hasattr(i, "path_shape") else 0 for i in []) or None
    tp = o3.TensorProduct(i1, i2, io, inst, shared_weights=False, internal_weights=False)
    x = torch.randn(n, i1.dim, device="cuda", dtype=torch.float64)
    e = torch.empty(0, i2.dim, device="cuda", dtype=torch.float64); w = torch.empty(0, tp.weight_numel, device="cuda", dtype=torch.float64)
    s = torch.empty(0, device="cuda", dtype=torch.int32)
    r = {}
    try:
        y = m(x, e, w, s, s); torch.cuda.synchronize(); r["fwd"] = {"ok": True, "shape": list(y.shape), "all_zero": bool((y == 0).all())}
    except Exception as ex:
        r["fwd"] = {"ok": False, "error": str(ex)[:300]}
    print(json.dumps(r)); json.dump(r, open(out, "w"))
elif mode == "refrep":
    import numpy as np
    sys.path.insert(1, f"{R}/scripts/flash_proto")
    from mace.calculators import MACECalculator
    from systems import get_system
    calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64", compute_stress=False)
    r = {}
    for name in ("H2O", "water192"):
        a0 = get_system(name); outs = []
        for _ in range(4):
            a = a0.copy(); a.calc = calc; calc.reset(); outs.append((a.get_potential_energy(), a.get_forces().copy()))
        r[name] = dict(bitwise_all=all(o[0] == outs[0][0] and np.array_equal(o[1], outs[0][1]) for o in outs),
                       max_dE=float(max(abs(o[0] - outs[0][0]) for o in outs)), max_dF=float(max(np.abs(o[1] - outs[0][1]).max() for o in outs)))
    print(json.dumps(r)); json.dump(r, open(out, "w"))
