"""Probe cuBLAS fp64 [M,3]@[3,3] rounding order (proto_graph shifts = S @ cell, pw = pos - w @ cell)."""
import torch
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
C = torch.tensor([[24.8, 0, 0], [7.7, 24.8, 0], [-4.2, 5.7, 26.1]], dtype=torch.float64, device=dev) + \
    torch.rand((3, 3), generator=g, device=dev, dtype=torch.float64) * 1e-3 * torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], device=dev)
def fma(a, b, c):  # exact-ish emulation via torch.addcmul is not fused; use float128-free trick: compute with mpmath? use CUDA kernel instead
    return torch.addcmul(c, a, b)
for M in [37, 1000, 32864, 39488, 300000]:
    S = torch.randint(-150, 150, (M, 3), generator=g, device=dev).double()
    ref = S @ C
    a0, a1, a2 = S[:, 0:1] * C[0], S[:, 1:2] * C[1], S[:, 2:3] * C[2]
    cands = {"(0+1)+2": (a0 + a1) + a2, "(0+2)+1": (a0 + a2) + a1, "0+(1+2)": a0 + (a1 + a2)}
    print(M, {k: int((v != ref).sum().item()) for k, v in cands.items()}, flush=True)
    # same op at two different M: is row result shape-independent?
    ref2 = S[:M // 2] @ C
    print("  prefix-consistent:", bool(torch.equal(ref2, ref[:M // 2])))
