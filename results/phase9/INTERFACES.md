# Phase 9 interfaces: GPU-resident MD step for node-hybrid (coordinator-owned; agents do not edit)

Root R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study. `source $R/env.sh`; python: $R/venv-cueq/bin/python -I.
GPU job env: copy scripts/prof8/job.sh / scripts/validate7/job.sh (LD_LIBRARY_PATH to venv-cueq nvidia libs,
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, PATH=$R/venv-cueq/bin:$PATH for ninja). `srun -p h100 --gres=gpu:1 -c 8 -t <=01:00:00 bash <script>`;
one H100 per job; log SLURM_JOB_ID + nvidia-smi uuid/clocks; end all jobs. TIMING jobs: do not run two timing jobs at once
(other agents' correctness jobs on other GPUs are fine; note them). No pip installs anywhere. Do not modify any existing
src/*, venv*, scripts/<earlier dirs>/*, or results/logs of earlier phases. Read-only reuse of earlier code is encouraged (copy then edit).

Agents / dirs (each writes only scripts/<dir>, results/<dir>, logs/<dir>):
  A nl9   : GPU cell-list neighbour builder emitting dst-sorted edges + CSR directly
  B md9   : GPU-resident velocity-Verlet loop, dynamic correctness harness, NVE drift, MD throughput
  C int9  : profiling + FINAL INTEGRATION OWNER (runner/capture with external CSR, benchmarks, final tables)

Current best ("node-hybrid", Phase 8): scripts/prof8/cfg8.py build("nodehybrid+embed+node") = hybrid_gc.build_calc(conv="e3nn",
sc="cueq") + scripts/nodetp8_bwd/nodeconv.install + scripts/fuse7_embed/fused_embed.install + scripts/fuse7_node/fused_node.install,
run by scripts/graphed7/graphed_runner.py GraphedRunner (keyed CUDA graphs, padded edges E_cap = round_cap(1.2E, 64), recapture
at 1.25E) with GPU builder scripts/graph/proto_graph.py ProtoBuilder (O(N^2) brute force + nonzero() host sync).
nodeconv builds CSR (2 stable sorts + 2 searchsorted) once per model call (_CsrCache in nodeconv.py).
FlashTP hybrid baseline: graphed7 build_backend("hybrid", transforms=(fused_embed.install, fused_node.install)).
Stock: MACECalculator(model_paths=$R/models/MACE-OFF23_medium.model, device="cuda", default_dtype="float64").

## Graph conventions to preserve (verify against stock MACE get_neighborhood / proto_graph)
edge_index [2,E] int64: row0 = sender (j... check), row1 = receiver; MACE vector = pos[receiver] - pos[sender] + shifts
(check sign convention in mace.modules.utils.get_edge_vectors_and_lengths); shifts = unit_shifts @ cell; pairs with
d < r_max (5.0 A; read calc.r_max); self-edges removed only when shift == 0 (periodic self-images kept); neighbour list
symmetric. Positions may be outside the cell (MACE wraps? check); proto_graph replicates matscipy's convention exactly — use
it and stock get_neighborhood as references (compare sorted edge sets incl. unit_shifts, bitwise).

## Builder API (A)  scripts/nl9/cell_nl.py
  NL = CellNeighborList(r_max, pbc, cell, n_atoms, E_cap, device)        # static geometry per key
  out = NL.build(positions[N,3] fp64 cuda)  -> dict of tensors, all fixed shape for fixed (N, E_cap):
     edge_index [2,E_cap] int64 (dst-sorted, padding at the end), unit_shifts [E_cap,3], shifts [E_cap,3] fp64,
     rowptr_dst [N+1] int32 (rowptr_dst[N] = E real), perm_src [E_cap] (src-sorted order WITHOUT a sort, e.g. via the
     symmetric reverse-edge map), rowptr_src [N+1], n_edges (0-d device int), overflow (0-d device int/bool)
  - no host sync, no data-dependent allocation, CUDA-graph capturable; overflow (E > E_cap or cell-bin overflow) detected
    on device and NEVER silently drops neighbours (caller checks flag; see C).
  - padding edges: must contribute exactly 0 energy/force/stress through MACE (existing trick: self-edge with shift >= 2*r_max
    on the last node so dst-sorted order is preserved); document.
  - supported geometries: non-periodic (bounding box bins), periodic orthorhombic & triclinic with perpendicular widths
    >= r_max per bin (multiple images if box < 2 r_max? else fallback); unsupported -> raise -> caller uses proto_graph.
  - also: a function to produce the full MACE batch dict (same keys/dtypes as proto_graph.build) from out + static template.

## Integration (C)  scripts/int9/
  - nodeconv variant (copy) accepting an EXTERNAL CSR (static buffers written by the builder) so no sort runs in the model.
  - runner variant: step from GPU-resident positions (no host copy-in) -> NL.build -> model fwd+bwd -> forces/energy/stress,
    optionally the whole step (NL + model) captured in one CUDA graph; overflow handling: explicit, documented, counted.
  - API for B:  sim = GPUStep(config, atoms_template, ...);  E, F = sim.forces(pos_gpu)  (tensors on GPU, no host sync)
                sim.check() -> host-side overflow/recapture check (explicit sync; B counts how often it is called)
## MD (B)  scripts/md9/
  - velocity Verlet on GPU tensors (positions, velocities, masses), NVE; force provider pluggable: stock-eager (ASE-free
    reference using stock MACE batch), graphed node-hybrid with proto builder (existing), and C's GPUStep (new).
  - dynamic correctness: >=100 steps, sample identical coordinates every k steps and compare energy/forces/stress vs stock
    fp64 MACE eager (and vs existing builder) with the tolerances below; NVE total-energy drift for a stable trajectory.
## Tolerances (fp64): |dE|/atom <= 1e-11 eV; max|dF| <= 1e-11 eV/A; stress <= 1e-14 eV/A^3; neighbour sets bitwise equal.
## Systems: results/workloads/systems/*.extxyz (aspirin, org192, org648, water192, water648, wbox192, wbox648) + H2O +
  zero-edge (single C; far O pair) + larger: water clusters/boxes ~2000 and ~4000-5200 atoms (generate deterministically,
  e.g. scripts/flash_proto/systems.water_cluster(n_side=9/12) and the water_box() in scripts/graph/graph_profile.py with
  n_side 9/12, density 1.0) — save as extxyz in results/<dir>/systems/.
## Timing boundaries (report all three, same boundary across backends; median [p10-p90], >=10 warmups, >=50 reps)
  B1 energy+forces graph replay with prebuilt neighbour graph;
  B2 complete step from GPU-resident coordinates incl. neighbour construction + force evaluation (+ any sync it needs);
  B3 sustained MD throughput incl. integration + whatever host interaction is actually required (ms/step, atoms*steps/s).
  Backends: stock MACE (eager; ASE for B3), FlashTP hybrid graphed, node-hybrid (Phase 8) graphed, new (Phase 9).
  Record compile, first-capture, peak memory, capacity constraints separately.
