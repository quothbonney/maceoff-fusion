"""NVE VelocityVerlet. Usage: run_nve.py <backend e3nn|flash> <dt_fs> <nsteps> <save_every> <tag> [system]
Saves every step: Epot, Ekin. Every save_every steps: positions, velocities, forces."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from mdlib import *
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
backend, dt, nsteps, every, tag = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
name = sys.argv[6] if len(sys.argv) > 6 else "water192"
st = np.load(f"{OUT}/start_{name}.npz")
atoms = Atoms(numbers=st["numbers"], positions=st["positions"], pbc=False); atoms.set_masses(st["masses"])
atoms.set_velocities(st["velocities"])
calc, model, _ = make_calc(backend); atoms.calc = calc
epot, ekin, P, V, F, steps = [], [], [], [], [], []
def obs(step):
    epot.append(atoms.get_potential_energy()); ekin.append(atoms.get_kinetic_energy())
    if step % every == 0:
        P.append(atoms.get_positions()); V.append(atoms.get_velocities()); F.append(atoms.get_forces()); steps.append(step)
obs(0)
dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
torch.cuda.synchronize(); t0 = time.perf_counter(); tlast = t0
for s in range(1, nsteps + 1):
    dyn.run(1); obs(s)
    if s % 200 == 0:
        torch.cuda.synchronize(); t = time.perf_counter()
        print(f"step {s} Etot/atom={(epot[-1]+ekin[-1])/len(atoms):.8f} T={atoms.get_temperature():.1f} {(t-tlast)/200*1e3:.1f} ms/step", flush=True); tlast = t
torch.cuda.synchronize(); wall = time.perf_counter() - t0
np.savez_compressed(f"{OUT}/traj_{tag}.npz", epot=np.array(epot), ekin=np.array(ekin), positions=np.array(P),
                    velocities=np.array(V), forces=np.array(F), saved_steps=np.array(steps), dt_fs=dt, n_atoms=len(atoms))
info = dict(env_info(), backend=backend, dt_fs=dt, nsteps=nsteps, save_every=every, system=name, wall_s=wall,
            s_per_step=wall / nsteps, note="wall includes ASE overhead + per-step observer")
json.dump(info, open(f"{OUT}/traj_{tag}.json", "w"), indent=1, default=float); print(info)
