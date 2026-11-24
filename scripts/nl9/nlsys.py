"""nl9 systems + reference helpers (deterministic)."""
import os, sys
import numpy as np
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_proto"); sys.path.insert(0, f"{R}/scripts/workloads")
sys.path.insert(0, f"{R}/scripts/graph"); sys.path.insert(0, f"{R}/scripts/nl9")
import ase.io
from ase import Atoms
from ase.build import molecule

SYSDIR = f"{R}/results/nl9/systems"
STD = ["aspirin", "org192", "org648", "water192", "water648", "wbox192", "wbox648"]
BIG = {"wclus2187": ("cluster", 9), "wclus5184": ("cluster", 12), "wbox2187": ("box", 9), "wbox5184": ("box", 12)}


def gen_big(name):
    kind, n = BIG[name]
    if kind == "cluster":
        from systems import water_cluster
        return water_cluster(n)
    from systems2 import water_box
    return water_box(n)


def load(name):
    if name == "H2O":
        return molecule("H2O")
    if name == "single_C":
        return Atoms("C", positions=[[0, 0, 0]])
    if name == "far_pair":
        return Atoms("O2", positions=[[0, 0, 0], [20.0, 0, 0]])
    if name in BIG:
        p = f"{SYSDIR}/{name}.extxyz"
        if not os.path.exists(p):
            os.makedirs(SYSDIR, exist_ok=True)
            ase.io.write(p, gen_big(name))
        return ase.io.read(p)
    if name == "wbox648_tric":     # sheared (triclinic) cell, same fractional coordinates
        a = load("wbox648"); L = a.cell[0, 0]
        frac = a.get_scaled_positions(wrap=False)
        C = np.array([[L, 0, 0], [0.31 * L, L, 0], [-0.17 * L, 0.23 * L, 1.05 * L]])
        b = a.copy(); b.set_cell(C); b.positions = frac @ C
        return b
    if name == "wbox2187_tric":
        a = load("wbox2187"); L = a.cell[0, 0]
        frac = a.get_scaled_positions(wrap=False)
        C = np.array([[L, 0, 0], [0.2 * L, 0.98 * L, 0], [0.11 * L, -0.27 * L, L]])
        b = a.copy(); b.set_cell(C); b.positions = frac @ C
        return b
    if name == "wbox648_outside":  # atoms displaced by random integer cell vectors + some far away
        a = load("wbox648"); rng = np.random.default_rng(5)
        k = rng.integers(-3, 4, size=(len(a), 3)); k[::7] *= 40
        b = a.copy(); b.positions = a.positions + k @ np.array(a.cell)
        return b
    if name == "wbox648_tric_outside":
        a = load("wbox648_tric"); rng = np.random.default_rng(6)
        k = rng.integers(-2, 3, size=(len(a), 3))
        b = a.copy(); b.positions = a.positions + k @ np.array(a.cell) + 0.0
        return b
    return ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")


def boundary_pairs(r_max, periodic, seed=0):
    """Pairs at r_max + delta (delta in +-1e-12, +-1e-13, +-2e-15, 0) along axes and random directions.
    periodic: cubic L=62 cell; every other pair straddles the x=0 face (partner at negative x, outside the cell)."""
    rng = np.random.default_rng(seed)
    deltas = [-1e-12, 1e-12, 0.0, -2e-15, 2e-15, -1e-13, 1e-13]
    pos = []
    for k, dlt in enumerate(deltas * 3):
        base = np.array([3.0 + 12.0 * (k % 5), 3.0 + 12.0 * ((k // 5) % 5), 3.0 + 12.0 * (k // 25)])
        if k < len(deltas):
            u = np.eye(3)[k % 3]
        else:
            u = rng.normal(size=3); u /= np.linalg.norm(u)
        if periodic and k % 2 == 1:
            base[0] = 0.5; u = u.copy(); u[0] = -abs(u[0]) if abs(u[0]) > 0 else u[0]
            if k < len(deltas):
                u = -np.eye(3)[0]
        pos.append(base); pos.append(base + (r_max + dlt) * u)
    a = Atoms("O" * len(pos), positions=np.array(pos))
    if periodic:
        a.set_cell([62.0, 62.0, 62.0]); a.set_pbc(True)
    return a


def canon(ei, us, sh):
    """Sort edges by (sender, receiver, S) -> numpy arrays."""
    ei = np.asarray(ei); us = np.asarray(us); sh = np.asarray(sh)
    o = np.lexsort((us[:, 2], us[:, 1], us[:, 0], ei[1], ei[0]))
    return ei[:, o], us[o], sh[o]


def stock_nl(atoms, r_max):
    from mace.data.neighborhood import get_neighborhood
    pbc = tuple(bool(x) for x in atoms.pbc)
    ei, sh, us, cell = get_neighborhood(atoms.positions, r_max, pbc=pbc,
                                        cell=np.array(atoms.cell) if any(pbc) else np.array(atoms.cell))
    return ei, us, sh, cell
