"""md9 start states (protocol of results/md/PROTOCOL.md step 1, extended).
For each system: geometry -> (30 FIRE steps, stock fp64 MACECalculator, only if not already relaxed) ->
MaxwellBoltzmann 300 K (np.random.RandomState(20261109)) -> Stationary -> ZeroRotation (clusters only).
water192/water648 reuse results/md/start_*.npz (Phase 4 start states) bit-for-bit.
Derived systems:
  water192_iso : water192 start + one O atom placed 30 A beyond the cluster's +x extent (empty neighbourhood, r_max 5),
                 velocity 0 (stays isolated; its force must be exactly 0)
  org192_rev   : org192 start with atom order reversed (species-order difference; same physical state)
  water2187 / wbox2187 / water5184 / wbox5184 : scripts/flash_proto/systems.water_cluster(9/12, spacing 3.1, seed 0),
                 scripts/graph/graph_profile.water_box(9/12, density 1.0, seed 1) (copied here), 30 FIRE (stock).
Output results/md9/systems/start_<name>.npz (positions, velocities[ASE units], numbers, masses, cell, pbc) + .extxyz.
usage: prep9.py name[,name...]"""
import os, sys, json, time
R = os.environ["R"]
import numpy as np
import ase.io
from ase import Atoms
from ase.build import molecule
from ase.optimize import FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
OUT = f"{R}/results/md9/systems"; os.makedirs(OUT, exist_ok=True)
SEED = 20261109


def water_cluster(n_side=6, spacing=3.1, seed=0):   # copy of scripts/flash_proto/systems.water_cluster
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


def water_box(n_side=6, density=1.0, seed=1):         # copy of scripts/graph/graph_profile.water_box
    nmol = n_side ** 3
    vol = nmol * 18.01528 * 1.66053906660e-24 / density * 1e24
    L = vol ** (1 / 3); sp = L / n_side
    a = water_cluster(n_side, spacing=sp, seed=seed)
    a.positions += sp / 2
    a.set_cell([L, L, L]); a.set_pbc(True)
    return a


_calc = None
def calc():
    global _calc
    if _calc is None:
        from mace.calculators import MACECalculator
        _calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64")
    return _calc


def save(name, atoms, info):
    np.savez(f"{OUT}/start_{name}.npz", positions=atoms.get_positions(), velocities=atoms.get_velocities(),
             numbers=atoms.numbers, masses=atoms.get_masses(), cell=np.array(atoms.cell), pbc=atoms.pbc)
    ase.io.write(f"{OUT}/{name}.extxyz", atoms)
    info.update(name=name, n_atoms=len(atoms), pbc=[bool(x) for x in atoms.pbc], T_start_K=atoms.get_temperature(),
                slurm_job=os.environ.get("SLURM_JOB_ID"))
    json.dump(info, open(f"{OUT}/start_{name}.json", "w"), indent=1, default=float); print(info, flush=True)


def load_start(name):
    st = np.load(f"{OUT}/start_{name}.npz")
    a = Atoms(numbers=st["numbers"], positions=st["positions"], cell=st["cell"], pbc=st["pbc"])
    a.set_masses(st["masses"]); a.set_velocities(st["velocities"]); return a


def thermalize(a, relax, info, name):
    if relax:
        a.calc = calc(); t0 = time.time()
        info["fmax_before"] = float(np.abs(a.get_forces()).max())
        FIRE(a, logfile=f"{R}/logs/md9/prep_fire_{name}.log").run(fmax=0.0, steps=30)
        info["fmax_after"] = float(np.abs(a.get_forces()).max()); info["fire_s"] = time.time() - t0
        info["fire_steps"] = 30; a.calc = None
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.RandomState(SEED))
    Stationary(a)
    if not any(a.pbc):
        ZeroRotation(a)
    info["mb_seed"] = SEED
    return a


def make(name):
    if name in ("water192", "water648"):
        st = np.load(f"{R}/results/md/start_{name}.npz")
        a = Atoms(numbers=st["numbers"], positions=st["positions"], pbc=False); a.set_masses(st["masses"])
        a.set_velocities(st["velocities"])
        return save(name, a, dict(src=f"results/md/start_{name}.npz (Phase 4 start: 30 FIRE e3nn + MB300 seed {SEED})"))
    if name in ("aspirin", "org192", "org648", "wbox192", "wbox648"):
        a = ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")
        info = dict(src=f"results/workloads/systems/{name}.extxyz (already 30 FIRE e3nn fp64)")
        return save(name, thermalize(a, False, info, name), info)
    if name == "water192_iso":
        a = load_start("water192"); x = a.positions[:, 0].max() + 30.0; c = a.positions.mean(0)
        b = a + Atoms("O", positions=[[x, c[1], c[2]]]); v = np.vstack([a.get_velocities(), np.zeros((1, 3))])
        b.set_masses(np.concatenate([a.get_masses(), [Atoms("O").get_masses()[0]]])); b.set_velocities(v)
        return save(name, b, dict(src="water192 start + isolated O 30 A beyond +x extent, v=0"))
    if name == "org192_rev":
        a = load_start("org192"); idx = np.arange(len(a))[::-1]
        b = Atoms(numbers=a.numbers[idx], positions=a.positions[idx], cell=a.cell, pbc=a.pbc)
        b.set_masses(a.get_masses()[idx]); b.set_velocities(a.get_velocities()[idx])
        return save(name, b, dict(src="org192 start, atom order reversed"))
    gen = {"water2187": lambda: water_cluster(9), "wbox2187": lambda: water_box(9),
           "water5184": lambda: water_cluster(12), "wbox5184": lambda: water_box(12)}
    a = gen[name]()
    info = dict(src=f"{name}: water_cluster(n_side,3.1,seed0) / water_box(n_side,1.0,seed1)")
    return save(name, thermalize(a, True, info, name), info)


if __name__ == "__main__":
    for n in sys.argv[1].split(","):
        make(n)
