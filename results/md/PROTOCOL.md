# NVE validation: MACE-OFF23 medium, e3nn vs FP64-CG FlashTP (2026-11-09)

Scripts: scripts/md/{mdlib,prep,run_nve,matched,analyze}.py, job wrapper scripts/md/job.sh. Logs: logs/md/. Data: results/md/.
Env: $R/venv (torch 2.11.0+cu128, mace-torch 0.3.17, ase 3.29.0), fp64 model, H100 80GB, 1 GPU per Slurm job.
FlashTP imported from src/flashTP-fp64cg (asserted in mdlib.py), adapter scripts/flash_proto/flashtp_adapter.install (both layers).

1. Start state (prep.py, job 981 / 987): get_system(name) -> 30 FIRE steps with e3nn (log logs/md/prep_fire_*.log;
   water192 Fmax 2.89 -> 0.35 eV/A) -> MaxwellBoltzmannDistribution(300 K, np.random.RandomState(20261109)) -> Stationary
   -> ZeroRotation. Saved to start_<sys>.npz (positions, velocities, numbers, masses); both backends start from it.
   Start T = 322 K (water192), 296 K (water648).
2. NVE (run_nve.py): ASE VelocityVerlet, non-periodic, MACECalculator neighbour list rebuilt every step (r_max 5.0 A).
   Epot/Ekin stored every step; positions/velocities/forces every k steps. Runs (water192, 2.0 ps each):
   dt 0.25 fs x 8000 (k=20), dt 0.5 fs x 4000 (k=10; each backend run twice), dt 1.0 fs x 2000 (k=5);
   water648 dt 0.5 fs x 3000 (1.5 ps, k=10). Files traj_<tag>.npz/.json (json: job id, GPU uuid, s/step).
3. Matched frames (matched.py): for every saved frame, e3nn (twice, calc.reset) and FlashTP on exactly that geometry;
   |dF| max, RMS dF, |dE|/atom, plus e3nn repeat and stored-vs-recomputed force. Output matched_<tag>.npy
   (cols: step, max|dF|, rmsdF, |dE|/atom, e3nn-repeat max|dF|, e3nn-repeat |dE|/atom, max|F|, stored-vs-reeval) + json.
4. Divergence (analyze.py): position RMSD between trajectories per saved frame; e3nn-vs-e3nn and flash-vs-flash reruns
   as the chaos control; ln(RMSD) linear fit over RMSD in [1e-13, 1e-3] A.
5. Energy: Etot/atom per step; drift = linear-fit slope over the full run (eV/atom/ps); uncertainty = std of slopes of
   4 equal blocks; RMS fluctuation = std of residual about linear fit (full run and first 1 ps).
   Neighbour list: pairs with d < 5.0 A (scipy cKDTree) per saved frame; directed edges = 2 x pairs.
Summary: summary.json; matched_*.json.

## Results (water192 unless noted; per atom)
Matched frames (all runs, 401 frames each; water648 301): FlashTP-vs-e3nn max|dF| <= 6.7e-15 eV/A, RMS dF <= 9.1e-16,
|dE| <= 3.0e-13 eV/atom; e3nn-vs-e3nn repeat on same geometry: max|dF| <= 4.3e-15, |dE| <= 3.0e-13, never bitwise equal.
Drift (eV/atom/ps) / RMS fluct (eV/atom):
 dt 0.25: e3nn -1.33e-6 / 5.97e-6   flash -1.35e-6 / 5.97e-6   (block std ~1.3e-6)
 dt 0.50: e3nn -5.99e-6, -6.02e-6 (rerun) / 2.35e-5   flash -5.92e-6, -5.90e-6 / 2.33e-5   (block std ~1.7e-6)
 dt 1.00: e3nn -2.79e-5 / 1.06e-4   flash -2.94e-5 / 1.07e-4   (block std 1e-5..1.7e-5)
 water648 dt 0.5 (1.5 ps): e3nn -9.02e-6 / 1.59e-5   flash -8.97e-6 / 1.60e-5
Divergence: all pairs (incl. e3nn-vs-e3nn) first differ at the first saved frame by ~1e-15 A, grow with lambda 17-20 /ps
(water648: 23.5 /ps), reach ~1e-7..1e-6 A at 1 ps and 0.3-1 A at 2 ps.
