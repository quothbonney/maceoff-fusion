"""Prototype GPU graph builder for MACECalculator (does NOT modify MACE).

build(atoms) -> dict with the same keys/dtypes/shapes/values as stock
  calc._clone_batch(calc._atoms_to_batch(atoms)) -> float cast -> .to_dict()
except that edges come out in a different (deterministic) order.

Changes vs stock:
  * static per-config tensors (node_attrs, atomic_numbers, batch, ptr, head, pbc, all zero/one
    property & weight tensors) are taken once from a stock batch and cached on the GPU, keyed on
    (atomic numbers, pbc, head); reused across calls (MD steps).
  * positions + cell + rcell + volume packed into ONE pinned float64 buffer -> one H2D copy.
    cell/rcell/volume computed on CPU with the exact same numpy/torch ops as stock (bitwise equal).
  * neighbour list: torch brute force on GPU (all pairs x periodic images), float64, d < r_max,
    self-edges with zero shift removed, unit shifts referred to the caller's (unwrapped) positions
    exactly like matscipy's ijS convention: D = pos[j] - pos[i] + S @ cell.
Falls back (raises) when atoms carry per-config properties the stock path would read (charges, info keys).
"""
import numpy as np
import torch
from mace.data import neighborhood as NB

DYN = ("positions", "edge_index", "shifts", "unit_shifts", "cell", "rcell", "volume")


def mace_cells(positions, pbc, cell, cutoff):
    """Exact replica of the cell logic in mace.data.neighborhood.get_neighborhood (no matscipy).
    Returns (returned_cell, extended_cell) as numpy float64."""
    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)
    search_directions = NB._aperiodic_search_directions(pbc, cell)
    search_positions = np.array(positions, dtype=float, copy=True)
    extended_cell = np.array(cell, dtype=float, copy=True)
    for dim, direction in search_directions.items():
        projection = search_positions @ direction
        extent = projection.max() - projection.min()
        extended_cell[dim, :] = (extent + 2 * cutoff + 1) * direction
        search_positions -= (projection.min() - cutoff) * direction
    if any(pbc):
        cell = np.array(cell, dtype=float, copy=True)
        for dim in range(3):
            if not pbc[dim] and not cell[dim].any():
                cell[dim] = extended_cell[dim]
    else:
        cell = extended_cell
    return cell, extended_cell


class ProtoBuilder:
    def __init__(self, calc, device="cuda", max_pair_elems=60_000_000):
        self.calc = calc
        self.device = torch.device(device)
        self.cutoff = float(calc.r_max)
        self.cache = {}
        self.max_pair_elems = max_pair_elems
        self._pinned = None
        self._img_cache = {}

    # ---- static template --------------------------------------------------
    def _template(self, atoms):
        pbc = tuple(bool(x) for x in atoms.get_pbc())
        key = (atoms.numbers.tobytes(), pbc, self.calc.head)
        t = self.cache.get(key)
        if t is None:
            b = self.calc._clone_batch(self.calc._atoms_to_batch(atoms))
            d = b.to_dict()
            t = dict(keys=list(d.keys()), static={k: v for k, v in d.items() if k not in DYN}, pbc=pbc)
            self.cache[key] = t
        return t

    def _check_supported(self, atoms):
        c = self.calc
        for name, k in c.arrays_keys.items():
            if k in atoms.arrays and k not in ("positions", "numbers"):
                raise NotImplementedError(f"atoms.arrays[{k!r}] present")
        for name, k in c.info_keys.items():
            if k in atoms.info:
                raise NotImplementedError(f"atoms.info[{k!r}] present")
        if getattr(c, "use_compile", False) or c.pad_num_atoms or c.pad_num_edges:
            raise NotImplementedError("padding/compile path")

    # ---- GPU neighbour list -----------------------------------------------
    def _images(self, nrep):
        im = self._img_cache.get(nrep)
        if im is None:
            r = [torch.arange(-n, n + 1, dtype=torch.float64) for n in nrep]
            im = torch.cartesian_prod(*r).reshape(-1, 3).to(self.device)
            self._img_cache[nrep] = im
        return im

    def neighbour_list(self, pos, pbc, search_cell_np, search_cell, ret_cell, w=None):
        """pos [N,3] f64 cuda; search_cell / ret_cell [3,3] f64 cuda; w [N,3] f64 cuda integer wraps (pbc only).
        Returns edge_index, unit_shifts, shifts."""
        rc2 = self.cutoff * self.cutoff
        N = pos.shape[0]
        if not any(pbc):
            d = pos.unsqueeze(0) - pos.unsqueeze(1)                 # d[i,j] = pos_j - pos_i
            d2 = (d * d).sum(-1)
            mask = d2 < rc2
            mask.fill_diagonal_(False)
            ij = mask.nonzero()                                     # sorted by (i, j)
            edge_index = ij.t().contiguous()
            us = torch.zeros((ij.shape[0], 3), dtype=torch.float64, device=pos.device)
            return edge_index, us, us @ ret_cell
        # periodic: wrap into the cell along periodic axes, search images, refer shifts back
        inv = np.linalg.inv(search_cell_np)
        heights = 1.0 / np.linalg.norm(inv, axis=0)                 # plane spacings
        nrep = tuple(int(np.ceil(self.cutoff / heights[k])) if pbc[k] else 0 for k in range(3))
        img = self._images(nrep)                                    # [M,3]
        pw = pos - w @ search_cell
        imv = img @ search_cell                                     # [M,3]
        M = img.shape[0]
        out_m, out_i, out_j = [], [], []
        chunk = max(1, self.max_pair_elems // (N * N))
        for m0 in range(0, M, chunk):
            iv = imv[m0:m0 + chunk]
            d = pw.unsqueeze(0).unsqueeze(0) - pw.unsqueeze(1).unsqueeze(0) + iv[:, None, None, :]   # [m,i,j,3]
            d2 = (d * d).sum(-1)
            mask = d2 < rc2
            z = (img[m0:m0 + chunk] == 0).all(-1)                   # zero-image -> drop i==j
            if z.any():
                mz = torch.nonzero(z).flatten()
                mask[mz] &= ~torch.eye(N, dtype=torch.bool, device=pos.device)
            idx = mask.nonzero()
            out_m.append(idx[:, 0] + m0); out_i.append(idx[:, 1]); out_j.append(idx[:, 2])
        m = torch.cat(out_m); i = torch.cat(out_i); j = torch.cat(out_j)
        S = img[m] - w[j] + w[i]
        # order like matscipy-ish: by sender, then receiver (stable)
        key = i * N + j
        order = torch.argsort(key, stable=True)
        i, j, S = i[order], j[order], S[order]
        edge_index = torch.stack((i, j))
        return edge_index, S, S @ ret_cell

    # ---- build -------------------------------------------------------------
    def build(self, atoms, timers=None):
        self._check_supported(atoms)
        t = self._template(atoms)
        pbc = t["pbc"]
        pos_np = atoms.arrays["positions"]
        ret_cell, ext_cell = mace_cells(pos_np, pbc, np.array(atoms.get_cell()), self.cutoff)
        cell_t = torch.tensor(ret_cell, dtype=torch.float64)
        volume = torch.linalg.det(cell_t)
        rcell = 2 * torch.pi * torch.linalg.inv(cell_t.mT) if torch.abs(volume) > 0 else torch.zeros(3, 3, dtype=torch.float64)
        N = len(atoms)
        periodic = any(pbc)
        nw = N * 3 if periodic else 0
        n = N * 3 + nw + 9 + 9 + 9 + 1
        if self._pinned is None or self._pinned.numel() < n:
            self._pinned = torch.empty(max(n, 4096), dtype=torch.float64).pin_memory()
        buf = self._pinned
        buf[:N * 3].copy_(torch.from_numpy(pos_np).reshape(-1))
        o = N * 3
        if periodic:   # integer wraps along periodic axes, computed on CPU (O(N))
            frac = pos_np @ np.linalg.inv(ext_cell)
            wnp = np.where(np.array(pbc)[None, :], np.floor(frac), 0.0)
            buf[o:o + nw].copy_(torch.from_numpy(wnp).reshape(-1)); o += nw
        buf[o:o + 9].copy_(cell_t.reshape(-1)); o += 9
        buf[o:o + 9].copy_(rcell.reshape(-1)); o += 9
        buf[o:o + 9].copy_(torch.from_numpy(ext_cell).reshape(-1)); o += 9
        buf[o] = volume
        g = buf[:n].to(self.device, non_blocking=True)
        # pinned buffer reuse: wait for copy before next call can overwrite it
        pos = g[:N * 3].view(N, 3).clone()
        o = N * 3
        w_g = g[o:o + nw].view(N, 3) if periodic else None; o += nw
        cell_g = g[o:o + 9].view(3, 3); o += 9
        rcell_g = g[o:o + 9].view(3, 3); o += 9
        ext_g = g[o:o + 9].view(3, 3); o += 9
        vol_g = g[o:o + 1]
        edge_index, us, shifts = self.neighbour_list(pos, pbc, ext_cell, ext_g, cell_g, w_g)
        # nonzero() inside neighbour_list synchronises the stream, so the pinned buffer is free again here
        dyn = dict(positions=pos, edge_index=edge_index, shifts=shifts, unit_shifts=us,
                   cell=cell_g.clone(), rcell=rcell_g.clone(), volume=vol_g.clone())
        st = t["static"]
        return {k: (dyn[k] if k in dyn else st[k]) for k in t["keys"]}
