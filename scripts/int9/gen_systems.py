"""Deterministic int9 systems -> results/int9/systems/*.extxyz (+ manifest.json).
water clusters: scripts/flash_proto/systems.water_cluster(n_side) (seed 0, spacing 3.1); periodic water boxes:
scripts/graph/graph_profile.water_box(n_side, density 1.0, seed 1) (re-implemented identically, no import side effects)."""
import os, sys, json, shutil, hashlib
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_proto")
import numpy as np, ase.io
from ase import Atoms
from ase.build import molecule
from systems import water_cluster

def water_box(n_side=6, density=1.0, seed=1):   # identical to scripts/graph/graph_profile.py water_box
    nmol = n_side ** 3
    vol = nmol * 18.01528 * 1.66053906660e-24 / density * 1e24
    L = vol ** (1 / 3); sp = L / n_side
    a = water_cluster(n_side, spacing=sp, seed=seed)
    a.positions += sp / 2
    a.set_cell([L, L, L]); a.set_pbc(True)
    return a

out = f"{R}/results/int9/systems"; os.makedirs(out, exist_ok=True)
S = {}
for n in (4, 6, 9, 12):
    S[f"water{4*0+3*n**3}"] = water_cluster(n)
    S[f"wbox{3*n**3}"] = water_box(n)
S["org648"] = ase.io.read(f"{R}/results/workloads/systems/org648.extxyz")
S["aspirin"] = ase.io.read(f"{R}/results/workloads/systems/aspirin.extxyz")
S["H2O"] = molecule("H2O")
S["zeroC"] = Atoms("C", positions=[[0, 0, 0]])
S["farO2"] = Atoms("OO", positions=[[0, 0, 0], [0, 0, 20.0]])
man = {}
for k, a in S.items():
    p = f"{out}/{k}.extxyz"; ase.io.write(p, a, format="extxyz")
    man[k] = dict(N=len(a), pbc=[bool(x) for x in a.pbc], cell=np.asarray(a.cell).tolist(),
                  sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
    print(k, len(a), a.pbc, np.diag(a.cell))
json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
