"""Kernel-level parity of a FlashTP tree vs e3nn (fp64 reference) -- review items b, c.

usage: python -I kernel_parity.py <tree> <label> <outdir>
For each config/layer: inputs are drawn in fp64; reference = e3nn TensorProduct + scatter in fp64.
FlashTP is built with dtype in {fp32, fp64} and torch default dtype in {fp32, fp64} at construction time.
Quantities (example flow of example/flashTP_test.py): out, d(gelu(out).sum())/d{in1,in2,w} (create_graph),
and the double-backward grads of sum(gelu(first grads)). Reports max abs err and max abs err / max |ref|.
fp32 outputs are saved to <outdir> for cross-tree bitwise comparison; each fp32 case is run twice to measure
run-to-run (atomicAdd) nondeterminism.
"""
import os, sys, json
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
tree, label, outdir = sys.argv[1:4]
sys.path.insert(0, tree)
import torch, e3nn.o3 as o3
import flashTP_e3nn
assert flashTP_e3nn.__file__.startswith(tree), flashTP_e3nn.__file__
R = os.environ["R"]
os.makedirs(outdir, exist_ok=True)

CASES = [("mace/medium.json", 0), ("mace/medium.json", 1), ("sevennet/sevennet-l3i5.json", 1),
         ("nequip/4_32_5_p_sc.txt", 1)]
N_NODE, NBR = 64, 64


def load(fn, li):
    lines = [l for l in open(f"{R}/src/flashTP/ir_config/{fn}").read().split("\n") if l.strip()]
    tp = json.loads(lines[li])["tp"]
    return o3.Irreps(tp[0]), o3.Irreps(tp[1]), o3.Irreps(tp[2]), [tuple(x) for x in tp[3]]


def chain(f, in1, in2, w, src, dst):
    in1, in2, w = [t.detach().clone().requires_grad_() for t in (in1, in2, w)]
    out = f(in1, in2, w, src, dst)
    y = torch.nn.functional.gelu(out).sum()
    g = torch.autograd.grad(y, [in1, in2, w], create_graph=True)
    loss = sum(torch.nn.functional.gelu(x).sum() for x in g)
    loss.backward()
    return [out.detach()] + [x.detach() for x in g] + [in1.grad, in2.grad, w.grad]


NAMES = ["out", "d1_in1", "d1_in2", "d1_w", "d2_in1", "d2_in2", "d2_w"]
res = {"tree": tree, "label": label, "file": flashTP_e3nn.__file__, "cases": {}}
for fn, li in CASES:
    i1, i2, io, inst = load(fn, li)
    g = torch.Generator().manual_seed(1234)
    E = N_NODE * NBR
    src = torch.randint(0, N_NODE, (E,), generator=g).cuda().int()
    dst = torch.arange(N_NODE).repeat_interleave(NBR)[torch.randperm(E, generator=g)].cuda().int()  # unsorted dst
    torch.set_default_dtype(torch.float64)
    tp = o3.TensorProduct(i1, i2, io, inst, shared_weights=False, internal_weights=False).cuda()
    torch.set_default_dtype(torch.float32)
    in1 = torch.rand(N_NODE, i1.dim, generator=g, dtype=torch.float64).cuda()
    in2 = torch.rand(E, i2.dim, generator=g, dtype=torch.float64).cuda()
    w = torch.rand(E, tp.weight_numel, generator=g, dtype=torch.float64).cuda()

    def ref_f(a, b, c, s, d):
        m = tp(a[s.long()], b, c)
        return torch.zeros(N_NODE, m.shape[1], dtype=m.dtype, device=m.device).index_add_(0, d.long(), m)
    ref = chain(ref_f, in1, in2, w, src, dst)
    for dtype in (torch.float32, torch.float64):
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            ftp = flashTP_e3nn.uvu_TP(i1, i2, io, inst, device="cuda", dtype=dtype)
            torch.set_default_dtype(torch.float32)
            runs = [chain(ftp, in1.to(dtype), in2.to(dtype), w.to(dtype), src, dst) for _ in range(2)]
            got = runs[0]
            tag = f"{fn}:L{li}|dtype={str(dtype)[6:]}|default={str(default)[6:]}"
            d = {"path": type(ftp).__name__}
            for n, a, r in zip(NAMES, got, ref):
                err = float((a.double() - r).abs().max()); mx = float(r.abs().max())
                d[n] = {"max_abs": err, "rel_to_max": err / mx if mx else float("nan")}
            d["worst_rel"] = max(d[n]["rel_to_max"] for n in NAMES)
            d["repeat_bitwise_identical"] = all(torch.equal(a, b) for a, b in zip(runs[0], runs[1]))
            if dtype == torch.float32:
                torch.save([t.cpu() for t in got], f"{outdir}/{label}_{fn.replace('/', '_')}_L{li}_default{str(default)[6:]}.pt")
            res["cases"][tag] = d
            print(label, tag, d["path"], f"worst_rel={d['worst_rel']:.3e}",
                  " ".join(f"{n}={d[n]['rel_to_max']:.2e}" for n in NAMES), "repeat_bitwise", d["repeat_bitwise_identical"], flush=True)
            del ftp
json.dump(res, open(f"{outdir}/kernel_parity_{label}.json", "w"), indent=1)
