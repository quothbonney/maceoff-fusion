"""Deterministic workload systems for the MACE-OFF23 medium workload study.
Usage:  python -I systems2.py build [--relax]     -> writes results/workloads/systems/<name>.extxyz (+ manifest.json)
        from systems2 import load_system, NAMES    -> loads saved (pre-relaxed) structures
Builders are deterministic (numpy default_rng with fixed seeds). Pre-relaxation (--relax): e3nn fp64 MACECalculator,
ase FIRE, <=30 steps, fixed cell; applied to organic systems and periodic boxes only. Water clusters are NOT relaxed
(water192/water648 are bit-identical to scripts/flash_proto/systems.py water_cluster(4/6))."""
import os, sys, json, math
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.collections import s22
from ase.io import read, write

R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
OUT = f"{R}/results/workloads/systems"
MODEL = f"{R}/models/MACE-OFF23_medium.model"
R_MAX = 5.0


def _rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, dd = q
    return np.array([[a*a+b*b-c*c-dd*dd, 2*(b*c-a*dd), 2*(b*dd+a*c)],
                     [2*(b*c+a*dd), a*a-b*b+c*c-dd*dd, 2*(c*dd-a*b)],
                     [2*(b*dd-a*c), 2*(c*dd+a*b), a*a-b*b-c*c+dd*dd]])


def water_grid(n_side, spacing=3.1, seed=0, jitter=0.1):
    """Same RNG sequence as flash_proto/systems.py water_cluster: returns per-molecule (pos[3,3], grid_idx)."""
    rng = np.random.default_rng(seed)
    w = molecule("H2O"); base = w.positions - w.positions.mean(0)
    mols = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                Rm = _rot(rng)
                mols.append((base @ Rm.T + spacing * np.array([i, j, k]) + rng.normal(scale=jitter, size=3), (i, j, k)))
    return mols


def water_cluster_n(n_mol, spacing=3.1, seed=0):
    """Generalized water cluster: grid of side ceil(cbrt(n)), keep the n molecules whose grid point is closest to the
    grid centre (stable sort on exact integer squared distance, ties by generation order). For perfect cubes this
    keeps all molecules -> identical to systems.py water_cluster(n_side)."""
    n_side = int(math.ceil(round(n_mol ** (1 / 3), 9)))
    mols = water_grid(n_side, spacing, seed)
    c2 = (n_side - 1)  # 2*centre, integer arithmetic
    d2 = [sum((2 * g - c2) ** 2 for g in idx) for _, idx in mols]
    keep = sorted(np.argsort(np.array(d2), kind="stable")[:n_mol])
    pos = np.concatenate([mols[i][0] for i in keep])
    return Atoms(symbols="OHH" * n_mol, positions=pos, pbc=False)


def water_box(n_side, density=1.0, seed=1, jitter=0.1):
    """Periodic liquid-like water: n_side^3 randomly rotated H2O on a cubic grid filling a cube at `density` g/cm3."""
    n = n_side ** 3
    L = (n * 18.015 / 0.602214076 / density) ** (1 / 3)
    mols = water_grid(n_side, spacing=L / n_side, seed=seed, jitter=jitter)
    pos = np.concatenate([m[0] for m in mols]) + 0.5 * L / n_side
    return Atoms(symbols="OHH" * n, positions=pos, cell=[L, L, L], pbc=True)


def aspirin():
    """Hand-built aspirin (C9H8O4) from idealized bond lengths/angles (planar skeleton, methyl H tetrahedral);
    intended to be pre-relaxed with MACE-OFF. Not a reference geometry."""
    P = {}; sym = {}
    def put(n, s, xyz): P[n] = np.array(xyz, float); sym[n] = s
    def u(deg): return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0])
    for k in range(6):
        put(f"C{k+1}", "C", 1.39 * u(90 - 60 * k))
    # C1 carboxylic acid, C2 acetoxy ester, ring H on C3..C6
    put("C7", "C", P["C1"] + 1.48 * u(90))
    put("O8", "O", P["C7"] + 1.21 * u(150))          # C=O away from ester side
    put("O9", "O", P["C7"] + 1.34 * u(30))           # O-H toward ester side (intramolecular contact)
    put("H9", "H", P["O9"] + 0.97 * u(90))
    put("O10", "O", P["C2"] + 1.40 * u(30))
    put("C11", "C", P["O10"] + 1.36 * u(-30))
    put("O12", "O", P["C11"] + 1.20 * u(30))
    put("C13", "C", P["C11"] + 1.50 * u(-90))
    a = math.radians(109.47)
    for i, phi in enumerate((0, 120, 240)):
        d = np.array([math.sin(a - math.pi / 2) * 0 + 0.0, 0, 0])
        # tetrahedral H around the C11->C13 axis (axis = -y)
        ax = np.array([0, -1.0, 0]); p1 = np.array([1.0, 0, 0]); p2 = np.array([0, 0, 1.0])
        d = math.cos(math.pi - a) * ax + math.sin(math.pi - a) * (math.cos(math.radians(phi)) * p1 + math.sin(math.radians(phi)) * p2)
        put(f"H13{i}", "H", P["C13"] + 1.09 * d)
    for k in (3, 4, 5, 6):
        put(f"H{k}", "H", P[f"C{k}"] + 1.08 * u(90 - 60 * (k - 1)))
    names = list(P)
    at = Atoms(symbols=[sym[n] for n in names], positions=[P[n] for n in names])
    at.positions -= at.positions.mean(0)
    return at


ORG_POOL = ["CH3CONH2", "CH3SH", "CH3CH2OH", "CH3CN", "CH3Cl", "C6H6", "C5H5N", "C4H4S", "CH3COOH", "H2O",
            "HCF3", "C2H6SO", "CH3NO2", "aspirin"]


def _mol(name):
    return aspirin() if name == "aspirin" else molecule(name)


def organic_cluster(target_atoms, seed=7, density=1.0, dmin=2.0):
    """Neutral non-periodic organic cluster: molecules drawn (seeded) from ORG_POOL, randomly rotated, packed by
    random sequential addition in a sphere sized for `density` g/cm3; reject if any inter-molecular distance < dmin.
    Molecules are added until n_atoms >= target_atoms."""
    rng = np.random.default_rng(seed)
    seq, n = [], 0
    while n < target_atoms:
        m = _mol(ORG_POOL[int(rng.integers(len(ORG_POOL)))]); seq.append(m); n += len(m)
    mass = sum(m.get_masses().sum() for m in seq)
    rad = (3 * mass / 0.602214076 / density / (4 * math.pi)) ** (1 / 3)
    pos, sym = np.zeros((0, 3)), []
    for m in seq:
        p0 = m.positions - m.positions.mean(0)
        for attempt in range(100000):
            if attempt and attempt % 500 == 0: rad *= 1.03
            v = rng.normal(size=3); v *= rad * rng.random() ** (1 / 3) / np.linalg.norm(v)
            p = p0 @ _rot(rng).T + v
            if len(pos) == 0 or np.min(np.linalg.norm(pos[:, None, :] - p[None], axis=-1)) >= dmin:
                pos = np.concatenate([pos, p]); sym += m.get_chemical_symbols(); break
        else:
            raise RuntimeError("packing failed")
    return Atoms(symbols=sym, positions=pos, pbc=False), [m.get_chemical_formula() for m in seq]


# name -> (builder, relax?)
def builders():
    return {
        "aspirin": (lambda: aspirin(), True),
        "AT_WC": (lambda: s22["Adenine-thymine_Watson-Crick_complex"].copy(), False),   # s22 reference geometry
        "org192": (lambda: organic_cluster(192)[0], True),
        "org384": (lambda: organic_cluster(384)[0], True),
        "org648": (lambda: organic_cluster(648)[0], True),
        "water192": (lambda: water_cluster_n(64), False),
        "water288": (lambda: water_cluster_n(96), False),
        "water384": (lambda: water_cluster_n(128), False),
        "water480": (lambda: water_cluster_n(160), False),
        "water648": (lambda: water_cluster_n(216), False),
        "wbox192": (lambda: water_box(4), True),
        "wbox648": (lambda: water_box(6), True),
    }


NAMES = list(builders())


def load_system(name):
    return read(f"{OUT}/{name}.extxyz")


def edge_count(atoms, cutoff=R_MAX):
    sys.path.insert(0, "")
    from mace.data.neighborhood import get_neighborhood
    ei, *_ = get_neighborhood(atoms.positions, cutoff, pbc=tuple(bool(x) for x in atoms.pbc),
                              cell=np.array(atoms.cell) if atoms.pbc.any() else None)
    return int(ei.shape[1])


def _mind(at):
    from ase.geometry import get_distances
    _, D = get_distances(at.positions, cell=at.cell, pbc=at.pbc)
    D[np.diag_indices_from(D)] = 9e9
    return float(D.min())


if __name__ == "__main__" and sys.argv[1] == "build":
    relax = "--relax" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    calc = None
    if relax:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        from mace.calculators import MACECalculator
        from ase.optimize import FIRE
        calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False)
    sys.path.insert(0, f"{R}/scripts/flash_proto")
    from systems import water_cluster
    for name, (b, rel) in builders().items():
        at = b(); info = dict(min_dist_built=_mind(at))
        if name in ("water192", "water648"):
            ref = water_cluster({"water192": 4, "water648": 6}[name])
            info["identical_to_flash_proto_systems"] = bool(np.array_equal(ref.positions, at.positions)
                                                            and ref.get_chemical_symbols() == at.get_chemical_symbols())
        if rel and relax:
            at.calc = calc
            f0 = float(np.abs(at.get_forces()).max())
            opt = FIRE(at, logfile=None); opt.run(fmax=0.05, steps=30)
            info.update(relax="e3nn fp64 FIRE fmax=0.05 steps<=30", relax_steps=int(opt.nsteps),
                        fmax_before=f0, fmax_after=float(np.linalg.norm(at.get_forces(), axis=1).max()))
            at.calc = None
        elif rel:
            info["relax"] = "SKIPPED (no --relax)"
        else:
            info["relax"] = "none"
        info.update(formula=at.get_chemical_formula(), n_atoms=len(at), pbc=bool(at.pbc.any()),
                    cell=np.array(at.cell).diagonal().tolist() if at.pbc.any() else None,
                    n_edges_rmax5=edge_count(at), min_dist=_mind(at),
                    elements=sorted(set(at.get_chemical_symbols())))
        if at.pbc.any():
            info["density_g_cm3"] = float(at.get_masses().sum() / 0.602214076 / at.get_volume())
        write(f"{OUT}/{name}.extxyz", at, format="extxyz")
        manifest[name] = info
        print(name, json.dumps(info), flush=True)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
