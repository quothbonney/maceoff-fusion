"""Shared graph format for node-centric conv kernels (phase8 INTERFACES).

build_csr(edge_index, N) -> dict(perm_dst [E] int64 (edge ids sorted by dst, stable), rowptr_dst [N+1] int32,
                                 perm_src [E] int64 (edge ids sorted by src, stable), rowptr_src [N+1] int32)
GPU only, no host sync, CUDA-graph capturable: torch.sort (stable) + torch.searchsorted on fixed shapes (no bincount:
CUDA bincount computes max() on the host -> sync). Requires N < 2^31. Edges of node n (as dst): perm_dst[rowptr_dst[n]:rowptr_dst[n+1]].
E == 0 -> perms empty, rowptrs all zero. Indices assumed in [0, N) (not checked: checking needs a host sync).
"""
import torch


def build_csr(edge_index, N):
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index shape {tuple(edge_index.shape)}, expected [2, E]")
    dev = edge_index.device
    nodes = torch.arange(N + 1, device=dev, dtype=torch.int32)
    out = {}
    for name, row in (("dst", edge_index[1]), ("src", edge_index[0])):
        keys, perm = torch.sort(row.to(torch.int32), stable=True)  # int32 keys: half the radix passes (~35% faster)
        out[f"perm_{name}"] = perm  # int64
        out[f"rowptr_{name}"] = torch.searchsorted(keys, nodes, right=False, out_int32=True)
    return out
