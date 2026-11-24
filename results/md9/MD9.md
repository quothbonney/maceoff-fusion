# Phase 9 / agent B (md9): GPU-resident velocity Verlet, dynamic correctness, NVE drift, B3 MD throughput

Date: 2026-11-10. Node: slurm-h100-206-077. One H100 per job, SM 1980 MHz. Env: venv-cueq (torch 2.11.0+cu128), fp64 MACE-OFF23 medium.
Machine-readable results:
- `results/md9/summary9.json` (all tables)
- `dyn_<provider>_<sys>[_tag].json` (every sample)
- `traj_<tag>.{npz,json}`
- `b3_<job>.json`

Start states: `results/md9/systems/start_<sys>.{npz,json,extxyz}`.
Scripts are in `scripts/md9/`; logs are in `logs/md9/`.

## Verdict
* **Dynamic correctness: PASS, 0 failing samples in every run.**
  * Covers 11 systems (21-2187 atoms, clusters and periodic boxes, an isolated atom, reversed species order), 500-1000 MD steps each.
  * Three providers were tested:
    * graphed node-hybrid with the proto builder (Phase 8)
    * C's GPUStep with A's cell-list NL
    * C's GPUStep with the brute NL stub
  * Every sample is stock fp64 eager on bitwise-identical coordinates.
  * Worst values over all runs:
    * |dE|/atom 1.3e-12 eV (limit 1e-11)
    * max|dF| 8.4e-15 eV/A (limit 1e-11)
    * |dstress| 4.7e-17 eV/A^3 (limit 1e-14)
  * Neighbour sets (sender, receiver, unit shift) are bitwise equal to stock and to proto_graph at every sample where the provider exposes them.
  * A negative control (one directed edge dropped) fails all 5 samples: max|dF| 0.42 eV/A, edge sets unequal. The harness is sensitive.
* **NVE drift: the new provider matches stock within the block-slope noise, at both time steps.**
  * GPUStep-cell and graphed node-hybrid produce **bitwise-identical positions** over the full 2 ps (water192; wbox648 at dt 0.5 and 0.25).
  * Drift scales about as dt^2 (wbox648: dt 0.5 → 0.25 gives 4.1x for new, 3.4x for stock).
  * GPU VV with the stock provider reproduces Phase 4's ASE VelocityVerlet trajectory: RMSD 1.8e-16 A at the first saved frame (step 10). It then diverges at the chaotic rate of Phase 4's own e3nn rerun (lambda 19.9 vs 17.0 /ps). This validates units and the algorithm.
* **B3 (sustained MD, incl. integration + all required host interaction): GPUStep vs graphed node-hybrid (Phase 8 + proto)**:

| system | speedup |
|---|---|
| water192 | 1.33x |
| water648 | 1.17x |
| wbox648 | 1.38x |
| water2187 | 1.08x |
| wbox2187 | 1.74x |
| water5184 | 1.08x |
| wbox5184 | 2.77x |

  * Against the user-facing baseline (ASE VelocityVerlet + MACECalculator), GPUStep is 11.4x (water192) to 19.1x (wbox2187) faster.
  * GPUStep needs **0 host syncs per step**, plus 1 event sync per `check()` (every 10 steps = 0.1/step).
  * The Phase 8 provider needs 2 (clusters) / 4-30 (periodic, proto chunking) syncs per step. Stock needs about 38 and ASE 43.
  * Capturing the VV integration as graphs around C's step graph (gpustep_cg) gains at most 0.6% (9 us at water192): the integration is not a bottleneck.

## 1. Integrator and providers (`scripts/md9/vv.py`)
**Integrator.** `VelocityVerletGPU(provider, atoms, dt_fs, n_max, save_every, check_every)` uses ASE units (eV, A, amu; dt = dt_fs·ase.units.fs). x, p and m live on the GPU in fp64, with masses from ASE.
* The algorithm is ASE's momentum form: p += ½dt·F; x += dt·p/m; (E,F) = force_fn(x); p += ½dt·F.
* F is a static GPU buffer.
* Epot and Ekin are written into preallocated GPU buffers at a device-side step index, so recording causes no host sync.
* Snapshots are optional GPU clones.

**Overflow protocol for providers with `check()`.** Every `check_every` steps (default 10), `check()` runs: 1 explicit sync, counted. Then:
* if ok: checkpoint (x, p, F, idx), as device-to-device copies;
* if `ok=False` (C's overflow → NaN results): roll back to the last checkpoint and redo the steps on C's recaptured graph, up to 4 times.

There were 0 rollbacks in all runs: the 20% E_cap margin plus proactive recapture was never exceeded.

**Providers.** Interface: `provider(x) -> (E 0-d, F [N,3])`, plus `.syncs`, `.last_stress`, `.edges()`, and optionally `.check()`.

| provider | what | host interaction per step |
|---|---|---|
| (a) `StockEager` | stock fp64 model on the stock batch (`calc._atoms_to_batch`, matscipy NL on CPU). ASE-free reference | 1 explicit position D2H + stock path (torch sync-audit: 38 syncs/step) |
| (b) `GraphedProto` | `cfg8.build("nodehybrid+embed+node")` + graphed7 `GraphedRunner(builder="proto")` | 1 explicit position D2H into host Atoms (counted) + proto builder: H2D + nonzero; audit 2 (clusters), 4 (wbox648), 8 (wbox2187), 30 (wbox5184; chunked image loop) |
| (c) `GPUStepProv` | C's `gpu_step.GPUStep(calc, model, atoms, nl=MD9_NL)` with `common9.build("nodeext+embed+node")`. NL + model in one CUDA graph; positions D2D-copied into the graph buffer | 0 (audit 0.0); `check()` = 1 event sync every 10 steps |
| (c') `gpustep_cg` | (c) + `capture_split()`: pre-kick+drift and post-kick+record captured as 2 small graphs around C's graph | same as (c) |

A whole-step single graph (`VelocityVerletGPU.capture`) is implemented but cannot be used with C's provider: C's graph would have to be launched inside another capture. The split capture is what the gpustep_cg rows use.

The sync-audit counts come from `torch.cuda.set_sync_debug_mode("warn")` over 10 untimed steps. It does not see `cudaEventSynchronize`, hence the explicit counters.

## 2. Dynamic correctness (`scripts/md9/dyncheck.py`)
**Protocol.**
* Start states (`prep9.py`, job 1068):
  * water192 and water648: Phase 4 starts (30 FIRE + MB 300 K, RandomState(20261109)).
  * aspirin, org192, org648, wbox192, wbox648: workloads systems, already 30 FIRE.
  * 2187/5184-atom water clusters and boxes: the same generators as A's `results/nl9/systems`. The geometries are identical to theirs (≤2e-4 A from extxyz rounding). They then get 30 FIRE steps with stock.
  * All systems: MB 300 K with the same seed, Stationary, and ZeroRotation for clusters only.
* Derived systems:
  * `water192_iso`: one O atom placed 30 A beyond the cluster, v=0 → empty neighbourhood.
  * `org192_rev`: atom order reversed.
* dt 0.5 fs. 1000 steps for N ≤ 648, sampled every 5 steps (201 samples). 500 steps for 2187 atoms, sampled every 10 (51 samples).
* At each sample, the exact GPU positions the provider just used are copied to the host. Stock fp64 eager is evaluated on a fresh stock batch and compared:
  * |dE|/atom, max and RMS |dF|, Voigt stress (periodic; provider runs with stress=True);
  * sorted (sender, receiver, unit_shift) rows vs stock, and the proto_graph builder vs stock (bitwise);
  * directed edges entering/leaving between consecutive samples.

**Results.** All rows: 0 failing samples; provider edges == stock == proto at every sample.
* "steps / samples" uses 1000/201 for N ≤ 648 and 500/51 for 2187 atoms.
* "in / out" is directed edge rows entering/leaving between samples, summed over the run.
* "max over providers" means the max over graphed, GPUStep-cell and GPUStep-brute.

| system | N | pbc | steps / samples | edges (min-max) | in / out | worst dE/atom (eV), max over providers | worst max abs dF (eV/A) | worst dstress (eV/A^3) |
|---|---|---|---|---|---|---|---|---|
| aspirin | 21 | F | 1000/201 | 296-352 | 130/106 | 0 | 6.9e-15 | n/a |
| water192 | 192 | F | 1000/201 | 6582-7004 | 10468/10288 | 1.5e-13 | 5.3e-15 | n/a |
| water192_iso (empty neighbourhood) | 193 | F | 1000/201 | 6582-7004 | 10468/10288 | 1.5e-13 | 5.3e-15 | n/a |
| org192 | 196 | F | 1000/201 | 3648-4298 | 5450/4906 | 8.9e-13 | 8.4e-15 | n/a |
| org192_rev (species order) | 196 | F | 1000/201 | 3648-4298 | 5450/4906 | 1.3e-12 | 8.4e-15 | n/a |
| water648 | 648 | F | 1000/201 | 24330-26512 | 48706/50888 | 9.0e-14 | 6.0e-15 | n/a |
| wbox192 | 192 | T | 1000/201 | 9860-10632 | 18554/19138 | 1.5e-13 | 5.1e-15 | 4.7e-17 |
| wbox648 | 648 | T | 1000/201 | 33324-35746 | 74890/77084 | 9.0e-14 | 6.7e-15 | 4.3e-17 |
| water2187 | 2187 | F | 500/51 | 91836-99334 | 100726/108224 | 1.1e-13 | 6.0e-15 | n/a |
| wbox2187 | 2187 | T | 500/51 | 113368-120698 | 137984/145194 | 1.1e-13 | 5.3e-15 | 4.3e-17 |

Notes:
* **Isolated atom.** Force exactly 0 (provider and stock) and 0 edges touching it, for all three providers.
* **Reversed species order.** org192_rev produces the same edge-flux statistics as org192 and passes at the same tolerances.
* **wbox192 (L = 12.4 A < ~15 A) with `nl=cell`.** GPUStep takes its proto_graph fallback (`path=proto`, provider name `..._proto_proto`, job 1120). Forces, energy and stress pass. Provider edges are not exposed on that path, so the neighbour check there is proto vs stock only (equal). With `nl=brute`, wbox192 runs in the graph path and the edges are checked equal.
* **Positions outside the cell.** In the periodic runs, atoms leave the [0,L) box during MD (unwrapped coordinates). This is covered implicitly.
* **Controls.**
  * stock-vs-stock: 100 steps (job 1073), worst 1.5e-13 / 3.7e-15 / 2.4e-17. This is the harness floor.
  * Negative control: job 1088, 20 steps, `--inject drop_edge`. 5/5 samples fail, max|dF| 0.42 eV/A.
* **Heating in the less-relaxed starts.** The 648/2187-atom systems heat during NVE: water2187 302→494 K and wbox2187 →548 K in 0.25 ps, from potential energy left after only 30 FIRE steps. Etot/atom ranges stay ≤4e-4 eV, as in Phase 4.

## 3. NVE drift (`scripts/md9/drift9.py`, 2 ps per run, energy stats as `scripts/md/analyze.py`)
Drift = linear-fit slope over the full run; ± = std of 4 block slopes; RMS = residual std (per atom).

| system / dt | stock eager (GPU VV) | graphed node-hybrid + proto | GPUStep cell NL | GPUStep brute NL | Phase 4 (ASE VV) |
|---|---|---|---|---|---|
| water192 / 0.5 fs, 4000 steps | -5.78e-6 ± 1.8e-6 / RMS 2.32e-5 | -5.95e-6 ± 1.9e-6 / 2.30e-5 | -5.95e-6 ± 1.9e-6 / 2.30e-5 (bitwise-identical positions to graphed) | -5.95e-6 ± 1.9e-6 / 2.30e-5 | e3nn -5.99e-6 ± 1.7e-6, rerun -6.02e-6; flash -5.92e-6 / 2.35e-5 |
| wbox648 / 0.5 fs, 4000 steps | -8.54e-6 ± 4.2e-6 / 1.40e-5 | -1.16e-5 ± 4.7e-6 / 1.42e-5 | -1.16e-5 ± 4.7e-6 / 1.42e-5 (bitwise = graphed) | -8.56e-6 ± 3.3e-6 / 1.40e-5 | (water648 cluster, 1.5 ps: -9.02e-6 / 1.59e-5) |
| wbox648 / 0.25 fs, 8000 steps | -2.53e-6 ± 1.2e-6 / 3.27e-6 | -2.86e-6 ± 1.1e-6 / 3.29e-6 | -2.86e-6 ± 1.1e-6 / 3.29e-6 (bitwise = graphed) | (cancelled, see failures) | e3nn water192 dt 0.25: -1.33e-6 / 5.97e-6 |

* Per-row differences are 0.7σ (wbox648 dt 0.5) and 0.2σ (dt 0.25), using the combined block std. The trajectories are not identical (chaos, lambda about 20 /ps); identical trajectories were not demanded.
* RMS fluctuation scales 4.3x from dt 0.5 to dt 0.25 (dt^2 → 4).
* wbox648 heats from 296 to about 420 K over 2 ps (the start is not equilibrated; 30 FIRE only). This is the same for all providers.
* **Bitwise identity.** GPUStep-cell and graphed (proto + in-model CSR sort) give bitwise-identical positions over the entire 2 ps in all three runs. Epot differs only in the last bits (reduction order).
  * Reading: A's dst-sorted / within-row-sender-sorted edge order reproduces exactly the order the Phase 8 stable sort produced, so forces are bitwise equal.
  * GPUStep-brute differs from cell from frame 20 onward (different edge order → last-bit differences → chaos).
* **Neighbour-list churn** (ASE neighbour_list on saved frames): water192 has about 79.5k directed-edge flips between frames 10 steps apart, over 2 ps. wbox648 has about 525k (every 20 steps).
* **VV vs ASE (units/algorithm check).** Stock provider on water192 dt 0.5 vs Phase 4 `traj_e3nn_dt0.5` (ASE VelocityVerlet + MACECalculator, same start):

| t (ps) | RMSD, this run (A) | Phase 4 e3nn-vs-e3nn rerun (A) |
|---|---|---|
| 0.005 | 1.8e-16 | 1.6e-16 |
| 0.5 | 2.2e-11 | 8.6e-12 |
| 1.0 | 6.2e-7 | 1.7e-8 |
| 2.0 | 0.63 | 0.28 |

  The step-0 Epot difference is 1.5e-13 eV/atom.

## 4. B3 sustained MD throughput (`scripts/md9/b3bench.py`, under results/phase9/TIMING_LOCK)
**Method.**
* dt 0.5 fs, 10 warmup steps, then 200 timed steps in 4 blocks of 50, with `cuda.synchronize` only at block ends.
* The table shows the median of the 4 block means in ms/step; [min-max] is in `b3_*.json`. The block spread is ≤1% except for ASE/stock at 2187 (≤4%).
* GPUStep runs with check() every 10 steps (included in the timing).
* Stock = GPU VV + StockEager (no stress). ASE = `VelocityVerlet` + default `MACECalculator` (computes stress for periodic cells, the user-facing default).

**Jobs.** 1084 (stock, ase, graphed) and 1113 (graphed, gpustep, gpustep_cg; MD9_NL=cell). Graphed is repeated in both jobs: water192 2.117/2.124, wbox5184 66.67/66.78 ms (≤0.6%).

| system | N | stock VV | ASE (user baseline) | graphed node-hybrid (Ph8) + proto | GPUStep (cell NL) | gpustep_cg | new vs Ph8 | new vs ASE | GPUStep atom-steps/s | peak MiB (Ph8 → new) |
|---|---|---|---|---|---|---|---|---|---|---|
| water192 | 192 | 17.22 | 18.22 | 2.124 | 1.594 | 1.585 | 1.33x | 11.4x | 1.21e5 | 788 → 980 |
| water648 | 648 | 44.36 | 45.86 | 3.614 | 3.085 | 3.080 | 1.17x | 14.9x | 2.10e5 | 1337 → 1531 |
| wbox648 | 648 | 54.99 | 56.33 | 5.194 | 3.755 | 3.747 | 1.38x | 15.0x | 1.73e5 | 1920 → 1944 |
| water2187 | 2187 | 140.6 | 143.9 | 10.08 | 9.293 | 9.293 | 1.08x | 15.5x | 2.35e5 | 4220 → 4426 |
| wbox2187 | 2187 | 204.6 | 207.0 | 18.84 | 10.83 | 10.82 | 1.74x | 19.1x | 2.02e5 | 5222 → 5192 |
| water5184 | 5184 | 392.6 | 397.6 | 23.33 | 21.52 | 21.51 | 1.08x | 18.5x | 2.41e5 | 9638 → 9826 |
| wbox5184 | 5184 | 432.1 | 437.8 | 66.78 | 24.13 | 24.13 | 2.77x | 18.1x | 2.15e5 | 11162 → 11128 |

(graphed = job-1113 value; stock/ASE = job 1084.)

Host syncs per step (sync-audit / explicit):

| provider | sync-audit / explicit |
|---|---|
| stock | 38 / 1 D2H |
| ASE | 43 |
| graphed | 2 (clusters), 4 (wbox648), 8 (wbox2187), 30 (wbox5184) / 1 D2H |
| GPUStep | 0 / 0.1 (check) |

Stock and ASE peak memory: 1.2 GB (192) to 44 GB (wbox5184).

**Reading.**
* For clusters, the Phase 8 → GPUStep gain is the removed position D2H + proto build (about 0.5-1.8 ms/step). The model dominates at ≥2000 atoms.
* For periodic boxes, the O(N²·images) proto build with its chunked syncs dominated the Phase 8 step (wbox5184: 66.8 → 24.1 ms).
* Stock is about 97% of ASE (ASE adds stress for periodic systems plus Python overhead).

**Reproducibility (job 1129).** All five providers ran in one timing-locked job; no other job was running on the node. Values are ms/step.

| system | stock | ASE | graphed | GPUStep | gpustep_cg | new vs Ph8 / vs ASE |
|---|---|---|---|---|---|---|
| water192 | 16.98 | 18.03 | 2.102 | 1.583 | 1.577 | 1.33x / 11.4x |
| wbox648 | 54.62 | 55.96 | 5.165 | 3.753 | 3.743 | 1.38x / 14.9x |
| wbox2187 | 206.1 | 210.1 | 18.84 | 10.82 | 10.79 | 1.74x / 19.4x |

Every value is within 1.5% of jobs 1084/1113.

## Host-sync accounting (explicit, per MD step)
* stock: 1 position D2H, plus the stock batch path (CPU NL, pageable H2D: 38 synchronising CUDA calls per the torch audit).
* graphed + proto: 1 position D2H (counter `d2h_positions`, exactly 1.0/step in all runs), plus proto H2D + `nonzero()` (+ `z.any()` per image chunk for periodic cells).
* GPUStep: 0 per step. `check()` = 1 `cudaEventSynchronize` per 10 steps (counter `check`, 0.101/step in dyncheck incl. the initial check). Recaptures: 0 across all MD runs. Rollbacks: 0. The split-capture post graph was captured once per run (`n_post_captures`=1).
* The correctness harness itself syncs at every sample. Those syncs are outside the provider and not part of B3.

## Failures / caveats
* **GPUStep construction failed in the early smoke (job 1076-era).** The error was "Cannot copy between CPU and CUDA tensors during CUDA graph capture" in C's brute_nl. C fixed it before my full runs; nothing in md9 was affected.
* **dyncheck wbox192 with cell NL failed first (job 1105).** It raised KeyError in `sim.edges()` on C's proto fallback path. I fixed my adapter: `edges()` → None when `sim.path != "graph"`. Rerun (job 1120) passes.
* **Transient slowdowns, not reproduced.** Some drift runs ran 5-16x slower than B3 during about 05:31-05:48: gpustep water192 8.7 ms/step and wbox648 dt 0.5 59-80 ms/step (jobs 1098, 1107).
  * Reruns of the same commands (jobs 1124, 1127) ran at 1.60 / 3.77 ms/step and gave bitwise-identical energies. The cause is external: it overlapped many concurrent agent jobs.
  * Drift values are unaffected. B3 is taken only from the timing-locked jobs, whose graphed rows reproduce between 1084 and 1113 and Phase 8 (wbox648 step 5.35 ms in Phase 8 vs 5.17-5.19 here).
  * The brute-NL wbox648 dt 0.25 drift run (job 1098) was cancelled while in that slow state. The brute stub is dev-only; the cell row covers dt 0.25.
* **Concurrency during B3.** Other agents' and my own correctness jobs ran on other GPUs during the B3 jobs (squeue listed in the logs). This is allowed by INTERFACES; the repeat agreement is ≤0.6%.
* **Graphed drift chain (job 1077) ran while vv.py was being edited.** The edits were functionally identical for this provider (static F buffer); its trajectory is bitwise identical to the later GPUStep-cell runs, which used the final code.
* **Stock drift log ends with "exit code 127".** I overwrote `drift_chain.sh` while bash was still reading it after the last run. All 3 runs completed with exit=0.

## Jobs
* 1068 prep
* dyncheck:
  * graphed: 1072, 1074, 1075
  * stock control: 1073
  * negative control: 1088
  * gpustep-brute: 1095, 1096, 1097
  * gpustep-cell: 1104, 1105 (wbox192 error), 1106
  * gpustep-cell wbox192: 1120
* drift:
  * stock: 1076
  * graphed: 1077
  * gpustep-brute: 1098 (dt 0.25 cancelled)
  * gpustep-cell: 1107
  * rerun: 1127
  * debug: 1124
* B3: 1084, 1113, 1129 (repeat)

## Commands (from $R; `source env.sh`)
```
srun -p h100 --gres=gpu:1 -c 8 -t 00:40:00 bash scripts/md9/job.sh prep9.py water192,water648,aspirin,org192,org648,wbox192,wbox648,water192_iso,org192_rev,water2187,wbox2187,water5184,wbox5184
srun ... bash scripts/md9/job.sh dyncheck.py graphed water192,water192_iso,aspirin,org192,org192_rev --steps 1000 --every 5
srun ... bash scripts/md9/job.sh dyncheck.py graphed wbox192,wbox648,water648 --steps 1000 --every 5
srun ... bash scripts/md9/job.sh dyncheck.py graphed water2187,wbox2187 --steps 500 --every 10
MD9_NL=cell srun ... bash scripts/md9/job.sh dyncheck.py gpustep <same sets> --tag cell     (MD9_NL=brute ... --tag brute)
srun ... bash scripts/md9/job.sh dyncheck.py stock water192,wbox192 --steps 100 --every 5
srun ... bash scripts/md9/job.sh dyncheck.py graphed water192 --steps 20 --every 5 --inject drop_edge --tag negctrl_drop_edge
[MD9_NL=cell] srun ... bash scripts/md9/drift_chain.sh <stock|graphed|gpustep> [_cell|_brute]
srun ... bash scripts/md9/timing_job.sh b3bench.py stock,ase,graphed water192,water648,wbox648,water2187,wbox2187,water5184,wbox5184
MD9_NL=cell srun ... bash scripts/md9/timing_job.sh b3bench.py graphed,gpustep,gpustep_cg <same systems>
venv-cueq/bin/python -I scripts/md9/analyze9.py      (CPU; writes summary9.json, prints all tables)
```
