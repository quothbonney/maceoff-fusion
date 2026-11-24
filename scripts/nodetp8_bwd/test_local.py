"""Local op correctness (real captured inputs, both layers) + gradcheck + E=0 + bwd timing vs FlashTP-graphcap."""
import os, sys, json, time
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HB  # pins flashTP graphcap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
from ase.io import read
from ase import Atoms
import nodeconv as NC
import nodetp_bwd
from layer_cfg import reference_fwd
from csr import build_csr
from mace.tools.scatter import scatter_sum

FWD = sys.argv[1] if len(sys.argv) > 1 else "ref"
DO_TIME = "--time" in sys.argv
calc, info = HB.build_calc(conv="e3nn", sc="e3nn", check_indices=False)
model = calc.models[0]
dev = torch.device("cuda")
systems = {}
for nm in ["aspirin", "org192", "water192", "org648", "water648", "wbox648"]:
    systems[nm] = read(f"{R}/results/workloads/systems/{nm}.extxyz")
systems["H2O"] = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
systems["single_C"] = Atoms("C", positions=[[0, 0, 0]])
systems["far_O_pair"] = Atoms("O2", positions=[[0, 0, 0], [20, 0, 0]])

h = NC.install(model, fwd="ref")
convs = NC.node_convs(model)
captured = {}
for nm, at in systems.items():
    for c in convs:
        c.record = []
    at = at.copy(); at.calc = calc
    at.get_potential_energy()
    captured[nm] = [c.record[0] for c in convs]
for c in convs:
    c.record = None
NC.uninstall(model, h)
fwd_fn, fwd_name = NC._load_fwd(FWD)
print("fwd:", fwd_name, flush=True)


def stats(a, b):
    ma = (a - b).abs().max().item() if a.numel() else 0.0
    mr = b.abs().max().item() if b.numel() else 0.0
    nz = b.abs() < 1e-8
    return dict(maxabs=ma, rel=ma / mr if mr > 0 else (0.0 if ma == 0 else float("inf")), maxref=mr,
                n_nearzero=int(nz.sum()), maxabs_nearzero=((a - b)[nz].abs().max().item() if nz.any() else 0.0))


res = dict(fwd=fwd_name, local={}, gradcheck={}, timing={})
gen = torch.Generator(device=dev).manual_seed(1)
worst = {}
for nm, recs in captured.items():
    for li, (x, y, w, ei) in enumerate(recs):
        tp = model.interactions[li].conv_tp
        cfg = convs[li].cfg
        N, E = x.shape[0], ei.shape[1]
        g = torch.randn((N, cfg.d_out), dtype=torch.float64, device=dev, generator=gen)
        xr, yr, wr = (t.clone().requires_grad_() for t in (x, y, w))
        ref = scatter_sum(tp(xr[ei[0]], yr, wr), ei[1], dim=0, dim_size=N)
        rgx, rgy, rgw = torch.autograd.grad(ref, (xr, yr, wr), g, allow_unused=True)
        rgx = torch.zeros_like(x) if rgx is None else rgx
        rgy = torch.zeros_like(y) if rgy is None else rgy
        rgw = torch.zeros_like(w) if rgw is None else rgw
        csr = build_csr(ei, N)
        ct = (csr["perm_dst"], csr["rowptr_dst"], csr["perm_src"], csr["rowptr_src"])
        xm, ym, wm = (t.clone().requires_grad_() for t in (x, y, w))
        out = NC.NodeConvFunction.apply(xm, ym, wm, ei, ct, cfg, fwd_fn)
        gx, gy, gw = torch.autograd.grad(out, (xm, ym, wm), g)
        r = dict(N=N, E=E, out=stats(out.detach(), ref.detach()), gx=stats(gx, rgx), gy=stats(gy, rgy), gw=stats(gw, rgw))
        assert gy.shape == (E, 16) and gw.shape == (E, cfg.W)
        res["local"][f"{nm}/L{li}"] = r
        for k in ("out", "gx", "gy", "gw"):
            worst[k] = max(worst.get(k, 0.0), r[k]["rel"] if r[k]["maxref"] > 0 else r[k]["maxabs"])
            worst[k + "_abs"] = max(worst.get(k + "_abs", 0.0), r[k]["maxabs"])
        print(nm, li, N, E, {k: (f"{r[k]['maxabs']:.2e}", f"{r[k]['rel']:.2e}", r[k]["n_nearzero"],
                                 f"{r[k]['maxabs_nearzero']:.1e}") for k in ("out", "gx", "gy", "gw")}, flush=True)
res["worst"] = worst
print("WORST", worst, flush=True)

# gradcheck tiny shapes (fp64): analytic bwd (kernel) vs numeric fwd (reference_fwd)
torch.manual_seed(0)
for li, c in enumerate(convs):
    cfg = c.cfg
    N, E = 4, 9
    ei = torch.randint(0, N, (2, E), device=dev)
    ei = ei[:, torch.argsort(ei[0])]
    csr = build_csr(ei, N)
    ct = (csr["perm_dst"], csr["rowptr_dst"], csr["perm_src"], csr["rowptr_src"])
    x = torch.randn(N, cfg.d1, dtype=torch.float64, device=dev, requires_grad=True)
    y = torch.randn(E, cfg.d2, dtype=torch.float64, device=dev, requires_grad=True)
    w = torch.randn(E, cfg.W, dtype=torch.float64, device=dev, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda a, b, c_: NC.NodeConvFunction.apply(a, b, c_, ei, ct, cfg, reference_fwd),
                                  (x, y, w), eps=1e-6, atol=1e-8, rtol=1e-6, fast_mode=False)
    # E = 0
    ei0 = torch.zeros((2, 0), dtype=torch.int64, device=dev)
    c0 = build_csr(ei0, N)
    gx0, gy0, gw0 = nodetp_bwd.bwd(cfg, torch.randn(N, cfg.d_out, dtype=torch.float64, device=dev), x.detach(),
                                   y.detach()[:0], w.detach()[:0], ei0, c0)
    torch.cuda.synchronize()
    res["gradcheck"][f"L{li}"] = dict(gradcheck=bool(ok), E0_gx_zero=bool((gx0 == 0).all()), E0_shapes=[list(gy0.shape), list(gw0.shape)])
    print("gradcheck", li, res["gradcheck"][f"L{li}"], flush=True)


def tmed(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts = np.array(ts)
    return dict(median_us=float(np.median(ts)), p10=float(np.percentile(ts, 10)), p90=float(np.percentile(ts, 90)))


if DO_TIME:
    flash = [HB.FA.FlashConv(model.interactions[li].conv_tp, torch.float64, dev, check_indices=False) for li in range(2)]
    for nm in ["aspirin", "org192", "water192", "org648", "water648", "wbox648"]:
        for li, (x, y, w, ei) in enumerate(captured[nm]):
            cfg = convs[li].cfg
            N = x.shape[0]
            g = torch.randn((N, cfg.d_out), dtype=torch.float64, device=dev)
            csr = build_csr(ei, N)
            r = {}
            r["node_bwd_kernel"] = tmed(lambda: nodetp_bwd.bwd(cfg, g, x, y, w, ei, csr))
            r["build_csr"] = tmed(lambda: build_csr(ei, N))
            xf, yf, wf = (t.clone().requires_grad_() for t in (x, y, w))
            of = flash[li](xf, yf, wf, ei)
            r["flashtp_gc_bwd"] = tmed(lambda: torch.autograd.grad(of, (xf, yf, wf), g, retain_graph=True))
            r["flashtp_gc_fwd"] = tmed(lambda: flash[li](x, y, w, ei))
            # check flash grads agree with ours
            fg = torch.autograd.grad(of, (xf, yf, wf), g, retain_graph=True)
            mg = nodetp_bwd.bwd(cfg, g, x, y, w, ei, csr)
            r["ours_vs_flash_rel"] = [((a - b).abs().max() / b.abs().max()).item() for a, b in zip(mg, fg)]
            res["timing"][f"{nm}/L{li}"] = r
            print("time", nm, li, N, ei.shape[1], {k: (round(v["median_us"], 1), round(v["p10"], 1), round(v["p90"], 1))
                                                   if isinstance(v, dict) else v for k, v in r.items()}, flush=True)
tag = FWD + ("_time" if DO_TIME else "")
json.dump(res, open(f"{R}/results/nodetp8_bwd/local_{tag}.json", "w"), indent=1)
print("DONE")
