"""md9 NVE drift run.  usage: drift9.py <provider> <system> <dt_fs> <nsteps> <save_every> <tag>
GPU velocity Verlet (vv.py) from results/md9/systems/start_<system>.npz; Epot/Ekin every step (GPU buffer, copied at
the end); positions/velocities every save_every steps. Output results/md9/traj_<tag>.npz + .json (energy stats as in
scripts/md/analyze.py: linear-fit drift over the full run, std of 4 block slopes, RMS residual)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vv as V  # first
import numpy as np
import torch

prov_kind, name, dt, nsteps, every, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), \
    int(sys.argv[5]), sys.argv[6]
OUT = f"{V.R}/results/md9"


def energy_stats(epot, ekin, n, dt):     # copy of scripts/md/analyze.energy_stats
    et = (epot + ekin) / n
    t = np.arange(len(et)) * dt / 1000.0
    slope, icpt = np.polyfit(t, et, 1); resid = et - (slope * t + icpt)
    blk = np.array_split(np.arange(len(et)), 4)
    bs = [np.polyfit(t[b], et[b], 1)[0] for b in blk]
    w = t <= 1.0; s1, i1 = np.polyfit(t[w], et[w], 1)
    kB = V.units.kB
    return dict(dt_fs=dt, n_steps=len(et) - 1, t_ps=float(t[-1]), drift_eV_atom_ps=float(slope),
                block_slopes_eV_atom_ps=[float(x) for x in bs], block_slope_std=float(np.std(bs, ddof=1)),
                rms_fluct_eV_atom=float(resid.std()), rms_fluct_first1ps_eV_atom=float((et[w] - (s1 * t[w] + i1)).std()),
                max_minus_min_eV_atom=float(et.max() - et.min()), Etot0_eV_atom=float(et[0]),
                Etot_end_minus_0=float(et[-1] - et[0]), T_start_K=float(2 * ekin[0] / (3 * n * kB)),
                T_end_K=float(2 * ekin[-1] / (3 * n * kB)), T_mean_K=float(np.mean(2 * ekin / (3 * n * kB))))


atoms = V.load_start(name)
prov = V.make_provider(prov_kind, atoms, stress=False)
md = V.VelocityVerletGPU(prov, atoms, dt, nsteps, save_every=every,
                         check_every=10 if prov_kind == "gpustep" else 0)
torch.cuda.synchronize(); t0 = time.perf_counter(); tl = t0
for s in range(1, nsteps + 1):
    md.step()
    if s % 500 == 0:
        torch.cuda.synchronize(); t = time.perf_counter()
        e = (md.epot[s] + md.ekin[s]).item() / len(atoms)
        print(f"step {s} Etot/atom={e:.8f} {(t - tl) / 500 * 1e3:.2f} ms/step", flush=True); tl = t
torch.cuda.synchronize(); wall = time.perf_counter() - t0
ep, ek = md.energies(); S, st = md.snapshots()
np.savez_compressed(f"{OUT}/traj_{tag}.npz", epot=ep, ekin=ek, positions=S[:, 0], velocities=S[:, 1], saved_steps=st,
                    dt_fs=dt, n_atoms=len(atoms), numbers=atoms.numbers, cell=np.array(atoms.cell), pbc=atoms.pbc)
info = dict(V.env_info(), provider=prov.name, system=name, dt_fs=dt, nsteps=nsteps, save_every=every, wall_s=wall,
            ms_per_step=wall / nsteps * 1e3, provider_syncs=dict(prov.syncs), check_failures=md.check_fail,
            runner_stats=prov.runner.stats() if hasattr(prov, "runner") else None,
            energy=energy_stats(ep, ek, len(atoms), dt),
            note="wall includes snapshot clones every save_every steps and a sync every 500 steps (progress print)")
json.dump(info, open(f"{OUT}/traj_{tag}.json", "w"), indent=1, default=float)
print(json.dumps(info["energy"]), info["ms_per_step"], flush=True)
