"""Probe torch's fp64 reduction order for (d*d).sum(-1) with last dim 3, at the shapes proto_graph uses."""
import torch
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
for shape in [(4_000_000, 3), (1, 1, 2000, 2000, 3), (27, 300, 300, 3), (1, 300, 300, 3), (1, 20, 20, 3), (5, 5, 3)]:
    d = (torch.rand(shape, generator=g, device=dev, dtype=torch.float64) - 0.5) * 10
    q = d * d
    ref = q.sum(-1)
    a, b, c = q[..., 0], q[..., 1], q[..., 2]
    cands = {"(a+b)+c": (a + b) + c, "(a+c)+b": (a + c) + b, "a+(b+c)": a + (b + c)}
    print(shape, {k: int((v != ref).sum().item()) for k, v in cands.items()}, flush=True)
