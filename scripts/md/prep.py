"""Start state: water192 (or argv[1]) -> 30 FIRE steps (e3nn) -> MB 300 K (RandomState(20261109)) -> Stationary+ZeroRotation."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from mdlib import *
from ase.optimize import FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
name = sys.argv[1] if len(sys.argv) > 1 else "water192"; NFIRE = 30
atoms = get_system(name); calc, model, _ = make_calc("e3nn"); atoms.calc = calc
e0 = atoms.get_potential_energy(); f0 = np.abs(atoms.get_forces()).max()
opt = FIRE(atoms, logfile=f"{R}/logs/md/prep_fire_{name}.log"); opt.run(fmax=0.0, steps=NFIRE)
e1 = atoms.get_potential_energy(); f1 = np.abs(atoms.get_forces()).max()
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.RandomState(20261109))
Stationary(atoms); ZeroRotation(atoms)
np.savez(f"{OUT}/start_{name}.npz", positions=atoms.get_positions(), velocities=atoms.get_velocities(),
         numbers=atoms.numbers, masses=atoms.get_masses())
info = dict(env_info(), system=name, n_atoms=len(atoms), fire_steps=NFIRE, E_initial=e0, Fmax_initial=f0, E_after_fire=e1,
            Fmax_after_fire=f1, T_start_K=atoms.get_temperature(), mb_seed=20261109)
json.dump(info, open(f"{OUT}/start_{name}.json", "w"), indent=1, default=float); print(info)
