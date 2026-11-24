"""tile-size sweep for _seg_linear_kernel (fwd+bwd plans) on synthetic one-hot inputs; prints us per launch under graph replay."""
import os, sys, itertools, json
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6"); import hybrid_gc as HB
sys.path.insert(0, f"{R}/scripts/fuse7_node"); import fused_node as FN
import torch
calc, _ = HB.build_calc(conv="e3nn", sc="e3nn", dtype=torch.float64); m = calc.models[0]; FN.install(m)
ops = {"L0.out": (m.interactions[0].__dict__[FN._OPS]["out"], 2048), "L1.out": (m.interactions[1].__dict__[FN._OPS]["out"], 5120),
       "L1.sc": (m.interactions[1].__dict__[FN._OPS]["sc"], 512)}
res = {}
def t(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(10): g.replay()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(200): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / 200 * 1e3
for N in (21, 192, 648):
    z = torch.randint(0, 3, (N,), device="cuda"); A = torch.nn.functional.one_hot(z, 10).double()
    for (nm, (op, din)), (br, bu, bk, nw) in itertools.product(ops.items(), [(32,64,32,4),(16,32,32,4),(16,64,32,4),(32,32,32,4),(16,32,16,2),(32,32,16,4),(64,64,32,4),(16,16,32,2),(32,64,64,4),(64,32,32,4)]):
        FN.BLOCK_R, FN.BLOCK_U, FN.BLOCK_K, FN.NUM_WARPS = br, bu, bk, nw
        x = torch.randn(N, din, dtype=torch.float64, device="cuda")
        a = A if op.species else None
        y = op.fwd.run(x, a); g = torch.randn_like(y)
        try:
            res[f"{N}/{nm}/{br},{bu},{bk},{nw}"] = (round(t(lambda: op.fwd.run(x, a)), 2), round(t(lambda: op.bwd.run(g, a)), 2))
        except Exception as e:
            res[f"{N}/{nm}/{br},{bu},{bk},{nw}"] = repr(e)[:80]
        print(f"{N}/{nm}/{br},{bu},{bk},{nw}", res[f"{N}/{nm}/{br},{bu},{bk},{nw}"], flush=True)
json.dump(res, open(f"{R}/results/fuse7_node/tune.json", "w"), indent=1)
