"""int9 development stub with the nl9 CellNeighborList API (results/phase9/INTERFACES.md), O(N^2) brute force.
CUDA-graph capturable: no host sync, no data-dependent shapes (compaction by cumsum + scatter into E_cap+1 slots).
Edges are emitted dst-major (dst asc, then image index, then src asc) -> dst-sorted; real edges [0, E), padding [E, E_cap).
Padding edge: src = dst = N-1, unit_shift (m,0,0) with m = max(1, ceil(2 r_max/|a1|)) (periodic) -> |shift| >= 2 r_max;
non-periodic: unit_shift 0, shift (2 r_max, 0, 0).  rowptr_dst[N] = rowptr_src[N] = E_cap (padding covered).
perm_src: ONE stable sort of src (stub only; nl9 provides it without a sort).  Overflow: n_edges (true count, may exceed
E_cap) and overflow = n_edges > E_cap on device; dropped edges are never silently accepted (caller checks / poisons).
Convention (= proto_graph / matscipy): vector = pos[dst] - pos[src] + S @ cell, S integer; wraps w = floor(pos @ inv(cell))
along periodic axes (pbc all-True or all-False only)."""
import math
import numpy as np
import torch


class BruteNeighborList:
    def __init__(self, r_max, pbc, cell, n_atoms, E_cap, device="cuda", block_elems=40_000_000, **_):
        self.r = float(r_max); self.N = int(n_atoms); self.E_cap = int(E_cap); self.dev = torch.device(device)
        pbc = tuple(bool(x) for x in pbc)
        if any(pbc) and not all(pbc):
            raise NotImplementedError("mixed pbc")
        self.periodic = all(pbc)
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self.cell_np = cell
        if self.periodic:
            inv = np.linalg.inv(cell)
            h = 1.0 / np.linalg.norm(inv, axis=0)
            nrep = [int(np.ceil(self.r / h[k])) for k in range(3)]
            rr = [torch.arange(-n, n + 1, dtype=torch.float64) for n in nrep]
            self.img = torch.cartesian_prod(*rr).reshape(-1, 3).to(self.dev)
            self.cell = torch.tensor(cell, device=self.dev); self.inv = torch.tensor(inv, device=self.dev)
            self.imv = self.img @ self.cell
            m = max(1.0, math.ceil(2 * self.r / np.linalg.norm(cell[0])))
            pad_us = torch.tensor([m, 0.0, 0.0], dtype=torch.float64)
            pad_sh = pad_us.to(self.dev) @ self.cell
        else:
            self.img = torch.zeros((1, 3), dtype=torch.float64, device=self.dev); self.imv = self.img
            pad_us = torch.zeros(3, dtype=torch.float64); pad_sh = torch.tensor([2 * self.r, 0.0, 0.0], device=self.dev)
        self.M = self.img.shape[0]
        self.zero_img = int(((self.img == 0).all(1)).nonzero()[0, 0])  # setup only
        N, Ec = self.N, self.E_cap
        self.block = max(1, min(N, block_elems // max(1, self.M * N)))
        # padding templates (slot Ec is a dump slot for dropped/overflow entries)
        self.t_src = torch.full((Ec + 1,), N - 1, dtype=torch.int64, device=self.dev)
        self.t_us = pad_us.to(self.dev).expand(Ec + 1, 3).contiguous()
        self.t_sh = pad_sh.reshape(1, 3).expand(Ec + 1, 3).contiguous()
        self.nodes = torch.arange(N + 1, dtype=torch.int32, device=self.dev)
        self.ar = torch.arange(Ec, dtype=torch.int64, device=self.dev)
        self.false1 = torch.zeros(1, dtype=torch.bool, device=self.dev)
        self.self_idx = {}
        for d0 in range(0, N, self.block):   # (b, zero_img, d0+b) positions of zero-image self pairs per dst block
            B = min(N, d0 + self.block) - d0
            bi = torch.arange(B, device=self.dev)
            self.self_idx[d0] = (bi, torch.full_like(bi, self.zero_img), bi + d0)

    def build(self, positions):
        N, Ec, M, dev = self.N, self.E_cap, self.M, self.dev
        r2 = self.r * self.r
        pos = positions
        if self.periodic:
            w = torch.floor(pos @ self.inv)
            pw = pos - w @ self.cell
        else:
            w = None; pw = pos
        src_o = self.t_src.clone(); dst_o = self.t_src.clone(); us_o = self.t_us.clone()
        off = torch.zeros((), dtype=torch.int64, device=dev)
        for d0 in range(0, N, self.block):
            d1 = min(N, d0 + self.block); B = d1 - d0
            # vec[b, m, s] = pw[dst] + imv[m] - pw[src]   (dst = d0+b)
            v = pw[d0:d1, None, None, :] + self.imv[None, :, None, :] - pw[None, None, :, :]
            mask = (v * v).sum(-1) < r2                                   # [B, M, N]
            mask.index_put_(self.self_idx[d0], self.false1.expand(B))     # no zero-image self edge
            flat = mask.reshape(-1)
            idx = torch.cumsum(flat, 0) - 1 + off
            off = off + flat.sum()
            tgt = torch.where(flat & (idx < Ec), idx, torch.full_like(idx, Ec))
            lin = torch.arange(flat.numel(), device=dev)
            b = lin // (M * N); m = (lin // N) % M; s = lin % N
            src_o.scatter_(0, tgt, s)
            dst_o.scatter_(0, tgt, b + d0)
            if self.periodic:
                S = self.img[m] - w[b + d0] + w[s]                     # S = img - w[dst] + w[src]
                us_o.scatter_(0, tgt[:, None].expand(-1, 3), S)
        n_edges = off
        src, dst, us = src_o[:Ec], dst_o[:Ec], us_o[:Ec]
        # restore padding rows (dump slot Ec absorbed dropped entries; rows >= n_edges were never written)
        pad = self.ar >= n_edges
        src = torch.where(pad, self.t_src[:Ec], src); dst = torch.where(pad, self.t_src[:Ec], dst)
        us = torch.where(pad[:, None], self.t_us[:Ec], us)
        shifts = us @ self.cell if self.periodic else torch.where(pad[:, None], self.t_sh[:Ec], torch.zeros_like(us))
        edge_index = torch.stack((src, dst))
        rowptr_dst = torch.searchsorted(dst, self.nodes, out_int32=True)
        _, perm_src = torch.sort(src, stable=True)                          # stub only
        ssorted = src[perm_src]
        rowptr_src = torch.searchsorted(ssorted, self.nodes, out_int32=True)
        return dict(edge_index=edge_index, unit_shifts=us, shifts=shifts, rowptr_dst=rowptr_dst, perm_src=perm_src,
                    rowptr_src=rowptr_src, n_edges=n_edges, overflow=n_edges > Ec)
