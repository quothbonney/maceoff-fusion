"""Shared test systems. water_cluster(6) is the 648-atom cluster from scripts/inspect_mace_layers.py (identical code/seed)."""
import numpy as np
from ase import Atoms
from ase.build import molecule

def water_cluster(n_side=6, spacing=3.1, seed=0):
    rng = np.random.default_rng(seed)
    w = molecule("H2O"); base = w.positions - w.positions.mean(0)
    pos, sym = [], []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, dd = q
                Rm = np.array([[a*a+b*b-c*c-dd*dd, 2*(b*c-a*dd), 2*(b*dd+a*c)],
                               [2*(b*c+a*dd), a*a-b*b+c*c-dd*dd, 2*(c*dd-a*b)],
                               [2*(b*dd-a*c), 2*(c*dd+a*b), a*a-b*b-c*c+dd*dd]])
                pos.append(base @ Rm.T + spacing * np.array([i, j, k]) + rng.normal(scale=0.1, size=3))
                sym += w.get_chemical_symbols()
    return Atoms(symbols=sym, positions=np.concatenate(pos), pbc=False)

def get_system(name):
    if name == "H2O": return molecule("H2O")
    if name.startswith("water"):  # water192 / water648 / water1536
        n = {"water192": 4, "water648": 6, "water1536": 8}[name]; return water_cluster(n)
    raise KeyError(name)
