import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from layer_cfg import make_cfg, tp_edges, reference_fwd, reference_bwd
from csr import build_csr
R = os.environ["R"]
model = torch.load(f"{R}/models/MACE-OFF23_medium.model", map_location="cpu").double()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
res = {}
for li, blk in enumerate(model.interactions):
    tp = blk.conv_tp.to(dev)
    cfg = make_cfg(tp)
    E, N = 300, 40
    x = torch.randn(N, cfg.d1, dtype=torch.float64, device=dev)
    y = torch.randn(E, cfg.d2, dtype=torch.float64, device=dev)
    w = torch.randn(E, cfg.W, dtype=torch.float64, device=dev)
    ei = torch.randint(0, N, (2, E), device=dev)
    ref = tp(x[ei[0]], y, w)
    mine = tp_edges(cfg, x[ei[0]], y, w)
    d = (ref - mine).abs().max().item()
    # backward vs autograd
    xr, yr, wr = (t.clone().requires_grad_() for t in (x, y, w))
    out = torch.zeros(N, cfg.d_out, dtype=torch.float64, device=dev).index_add(0, ei[1], tp(xr[ei[0]], yr, wr))
    g = torch.randn_like(out)
    gx, gy, gw = torch.autograd.grad(out, (xr, yr, wr), g)
    ogx, ogy, ogw = reference_bwd(cfg, g, x, y, w, ei)
    of = reference_fwd(cfg, x, y, w, ei)
    res[li] = dict(nnz=int(cfg.cg_val.numel()), paths=len(cfg.paths), tp_maxabs=d, tp_rel=d / ref.abs().max().item(),
                   fwd_maxabs=(of - out).abs().max().item(),
                   gx=(gx - ogx).abs().max().item(), gy=(gy - ogy).abs().max().item(), gw=(gw - ogw).abs().max().item())
    print(li, res[li], [(P["l1"], P["l2"], P["l3"], P["path_weight"]) for P in cfg.paths])
csr = build_csr(ei, N)
ok = torch.equal(csr["rowptr_dst"].long().diff(), torch.bincount(ei[1], minlength=N)) and \
     torch.equal(ei[1][csr["perm_dst"]], ei[1].sort().values)
print("csr ok", ok, build_csr(ei[:, :0], N)["rowptr_src"].unique())
