# int9 INTEGRATION (Phase 9, agent C): GPU-resident whole-step CUDA graph (cell-list NL + node-hybrid, external CSR)

Machine-readable: results/int9/INTEGRATION.json (final table), final_A.json / final_B.json (B1/B2), b3_final_gpustep.json (B3),
prof_final.json (profiles), test_full_*.json + test_binovf.json (correctness). Baseline (before): results/int9/BASELINE.md.
Code md5s: results/int9/final_md5.txt (gpu_step.py a4ce5f49...). Hardware: H100 80GB HBM3, SM 1980 MHz, slurm-h100-206-077.

## Architecture
* `scripts/int9/nodeconv_ext.py` is a copy of nodetp8_bwd/nodeconv.py with an `ExtCsr` holder that all layers share.
  * The step object calls `ext.set(edge_index, N, perm_dst, rowptr_dst, perm_src, rowptr_src)` before the model call.
  * A layer uses those tensors only if it receives the identical `edge_index` object and the same N. Otherwise it falls back to the original sort-based `build_csr`.
  * `install`/`uninstall` restore the exact original `conv_tp` objects; this is tested.
  * Model config: `common9.build("nodeext+embed+node")`, i.e. node-hybrid + fused_embed + fused_node.
* `scripts/int9/gpu_step.py` provides `GPUStep`. Positions live in a static GPU buffer. **One CUDA graph** per key contains:
  1. nl9 `CellNeighborList.build` (3 launches; edges dst-sorted; `perm_src` from the reverse-edge map).
  2. `_spread_padding`, which moves the E_cap-E padding self-edges from node N-1 to nodes k mod N. It rebuilds both CSRs **without a sort**: O(E) elementwise work plus 2 scatters.
  3. `ExtCsr.set`, then MACE fwd + force [+ stress] backward.
  4. NaN poisoning, sticky device flags, and a captured D2H copy of `[overflow bits, max E, last E]` into pinned memory.
  * Per call there are **0 host syncs**: one D2D copy into the position buffer plus one graph launch. Writing `sim.positions` in place avoids even the copy.
  * Graph key: (N, numbers in order, pbc, cell bytes, dtype, stress, head). An LRU holds up to 4 graphs.
  * Capacities: E_cap = round_cap(1.2 E0, 64); nl9 bin_cap uses its default.
* API for md9: `E, F[, S] = sim.forces(pos_gpu)` (GPU views, overwritten by the next call), `sim.check()`, `sim.forces_checked(pos)`, `sim.stats()`, `sim.set_atoms(atoms)`.
* **Overflow policy:**
  * The NL flags overflow on the device (bit 1: E > E_cap, bit 2: bin full, bit 4: missing reverse edge). On any overflow the model is fed an **all-padding graph**, because the builder's CSR may be inconsistent: before this fix, a bin overflow caused an illegal address / device assert. E, F and S come back as **NaN**.
  * The flag is sticky until `check()`. `check()` costs 1 event sync, which is counted. If any step since the last check overflowed, it recaptures with E_cap = round_cap(1.25 max_E) and/or bin_cap = max(2x, 1.5 max_bin), and returns ok=False. The caller then rolls back to the state of its last ok check; md9's VV does this.
  * Proactive recapture happens when max_E > 0.98 E_cap; results stay valid.
  * Invalid results are never returned as valid: a caller that never checks gets NaN.
* **Fallbacks:**
  * nl="auto" uses the cell list. If it raises NotImplementedError (a periodic box with < 3 bins/dim, e.g. wbox192 at L = 12.4 A; mixed pbc), GPUStep uses the proto path: ProtoGPU O(N^2) with a nonzero sync outside the graph, then the graphed7 model graph with the in-model CSR sort. Mixed pbc goes to the host proto builder.
  * The O(N^2) capturable stub `brute_nl.py` is selected only by nl="brute". In a whole graph it measured 6.05 ms vs 2.38 ms for the proto path at wbox192.
* **FlashTP-hybrid small-system dispatch: not needed.** The same whole-step graph with the FlashTP-hybrid model ("GPUStep w/ FlashTP model" rows) is never faster in B2, and is not faster at aspirin either: 0.872 vs 0.814 ms. With the sort removed and no proto sync, node-hybrid GPUStep is now the fastest B2 at every size.

## Correctness (jobs 1144 [final code], 1143 [bin overflow]): 0 failures. Tolerances: dE/atom 1e-11 eV, dF 1e-11 eV/A, dsigma 1e-14 eV/A^3
| test | pass | samples | worst dE/atom eV | worst max dF eV/A | worst dsigma eV/A^3 | systems | edge sets == stock (bitwise) |
|---|---|---|---|---|---|---|---|
| bin_overflow_recapture | True | 4 | 0.0e+00 | 4.4e-15 | 0.0e+00 | water648, wbox648 | - |
| fallback_paths | True | 15 | 1.5e-13 | 7.1e-15 | 5.6e-17 | water192_mixed_pbc, wbox192 | - |
| zero_edge_tiny | True | 3 | 0.0e+00 | 1.0e-15 | 0.0e+00 | H2O, farO2, zeroC | - |
| species_order_atom_count_traps | True | 7 | 1.7e-13 | 8.5e-13 | 0.0e+00 | aspirin/rev/water192 interleaved | - |
| overflow_recapture | True | 2 | 1.5e-13 | 4.9e-15 | 0.0e+00 | water192, wbox648 | - |
| overflow_forces_checked | True | 2 | 1.5e-13 | 4.9e-15 | 0.0e+00 | water192, wbox648 | - |
| overflow_compression_walk | True | 12 | 1.5e-13 | 3.6e-13 | 0.0e+00 | water192, wbox648 | - |
| systems_walk_vs_stock | True | 98 | 1.1e-12 | 2.1e-14 | 7.0e-17 | aspirin, org648, water192, water2187, water5184, water648, wbox192, wbox2187, wbox5184, wbox648 | all |
| systems_walk_vs_nodehybrid_proto | True | 98 | 6.3e-13 | 0.0e+00 | 1.3e-17 | aspirin, org648, water192, water2187, water5184, water648, wbox192, wbox2187, wbox5184, wbox648 | - |
| repeat_same_coords_tol | True | 14 | 1.8e-13 | 0.0e+00 | 0.0e+00 | aspirin, org648, water192, water2187, water5184, water648, wbox192, wbox2187, wbox5184, wbox648 | - |
* systems_walk is 7 configurations per system: start + 6 random steps of 0.02 A rms, so the NL changes between steps.
  * Stress is included for every periodic system, at 192-5184 atoms.
  * At each step the result is compared with stock fp64 eager on the same coordinates.
  * At each step the neighbour set (sender, receiver, unit_shift), taken from the graph's NL buffers, is **bitwise equal** to stock get_neighborhood.
* Against node-hybrid + proto (Phase 8) on identical coordinates, **forces are bitwise identical** (max dF = 0). Energies differ by <= 6e-13 eV/atom; that comes from atomic summation order.
* Repeating the same coordinates gives identical forces. Energy can differ by 1 ulp of the total (2.9e-11 eV at water192, i.e. 1.5e-13 eV/atom) because the MACE readout scatter uses atomics.
* Overflow handling:
  * E_cap overflow (E_cap = E/2, and E_cap = 64): the first result is NaN, check() returns ok=False, recapture, then the result is correct.
  * Compression walk (x0.97 per step): proactive recapture (water192) and per-cell keys (wbox648).
  * Bin overflow (bin_cap 4 open / 8 periodic): NaN, then bin_cap 64, then correct.
* Traps: aspirin / reversed species / water192 interleaved, giving 3 keyed graphs. Zero edges: single C, O-O at 20 A. H2O: E = 6 in E_cap = 64.
* The external CSR is really used: sort_fallback_layer_calls = 0 inside GPUStep graphs.
  * The 8 radix-sort kernels that remain in the step graph come from MACE's own `index` backward (gather of positions[edge_index]); they are present in every backend.
* Not covered here: atoms ~3000 A outside the cell, a case nl9 reports separately (stress 1.1e-14 at the proto-vs-stock noise floor). The NVE drift and dynamic tests are md9's (results/md9/MD9.md; 0 failures, using this GPUStep).

## Final timing (two separate timing jobs: A = 1145, GPU-884d19b4; B = 1146, GPU-4b202516; TIMING_LOCK held; 10 warm-ups, 60 synced reps, median [p10-p90])
Boundaries (identical for all backends):
* **B1**: graphed backends: graph.replay() with the NL prebuilt. GPUStep: graph replay of the whole step, so it **includes** the NL; GPUStep B1 is therefore about B2. Stock: eager model call on a prebuilt batch.
* **B2**: complete step from GPU-resident coordinates.
  * Graphed P7/P8: ProtoGPU build (O(N^2), nonzero sync) + load + replay.
  * GPUStep: forces(pos_gpu) = D2D copy + replay.
  * Stock: calc.calculate(atoms) (stock has no GPU-resident path).
* **B3**: md9's b3bench (sustained NVE velocity Verlet on GPU, check() every 10 steps), run under int9's lock with the final GPUStep (job 1147, results/int9/b3_final_gpustep.json).
  * Stock B3 = ASE VelocityVerlet + MACECalculator from md9 jobs 1084/1129. "VV+stock" = md9's GPU VV with the stock eager provider.
* Stock at >= 2187 atoms: 2 warm-ups, 10 reps.

| system | N | E | backend | B1 ms runA [p10-p90] / runB | B2 ms runA [p10-p90] / runB | B3 ms/step (stock: ASE VV; others: b3bench job 1147) | B2 speedup vs node-hybrid / FlashTP-hybrid / stock | peak MiB |
|---|---|---|---|---|---|---|---|---|
| aspirin | 21 | 302 | stock | 11.606 [11.529-11.886] / 11.438 | 16.336 [16.297-16.401] / 16.210 | - |  | 113 |
| aspirin | 21 | 302 | FlashTP-hybrid | 0.679 [0.678-0.680] / 0.686 | 0.910 [0.907-0.914] / 0.912 | - |  | 143 |
| aspirin | 21 | 302 | node-hybrid (P8) | 0.754 [0.753-0.755] / 0.760 | 0.938 [0.936-0.941] / 0.937 | 1.366 (job 1147) |  | 77 |
| aspirin | 21 | 302 | GPUStep (P9) | 0.807 [0.806-0.807] / 0.807 | 0.814 [0.813-0.814] / 0.814 | 0.896 (job 1147) | 1.15x / 1.12x / 20.08x | 77 |
| aspirin | 21 | 302 | GPUStep w/ FlashTP model | 0.866 [0.865-0.869] / 0.872 | 0.872 [0.872-0.876] / 0.879 | - | 1.08x / 1.04x / 18.72x | 78 |
| water192 | 192 | 6634 | stock | 13.453 [13.398-13.475] / 13.182 | 18.096 [18.061-18.141] / 17.830 | 18.028 (job 1129); VV+stock 16.98 |  | 1132 |
| water192 | 192 | 6634 | FlashTP-hybrid | 2.188 [2.185-2.195] / 2.206 | 2.372 [2.367-2.377] / 2.384 | - |  | 312 |
| water192 | 192 | 6634 | node-hybrid (P8) | 1.517 [1.513-1.521] / 1.542 | 1.704 [1.697-1.712] / 1.723 | 2.133 (job 1147) |  | 296 |
| water192 | 192 | 6634 | GPUStep (P9) | 1.553 [1.547-1.556] / 1.577 | 1.560 [1.555-1.565] / 1.585 | 1.601 (job 1147) | 1.09x / 1.52x / 11.60x | 296 |
| water192 | 192 | 6634 | GPUStep w/ FlashTP model | 2.367 [2.363-2.371] / 2.389 | 2.377 [2.372-2.380] / 2.396 | - | 0.72x / 1.00x / 7.61x | 311 |
| wbox192 | 192 | 10588 | stock | 15.816 [15.798-15.946] / 15.789 | 22.005 [21.969-22.041] / 21.867 | - |  | 1623 |
| wbox192 | 192 | 10588 | FlashTP-hybrid | 2.948 [2.945-2.950] / 2.967 | 3.605 [3.592-3.621] / 3.603 | - |  | 443 |
| wbox192 | 192 | 10588 | node-hybrid (P8) | 1.730 [1.726-1.734] / 1.757 | 2.380 [2.370-2.395] / 2.392 | 2.784 (job 1147) |  | 423 |
| wbox192 | 192 | 10588 | GPUStep (P9) | 1.739 [1.734-1.750] / 1.764 | 2.414 [2.403-2.435] / 2.425 | 2.342 (job 1147) | 0.99x / 1.49x / 9.12x | 423 |
| wbox192 | 192 | 10588 | GPUStep w/ FlashTP model | 2.954 [2.951-2.957] / 2.972 | 3.630 [3.619-3.647] / 3.632 | - | 0.66x / 0.99x / 6.06x | 443 |
| org648 | 650 | 14856 | stock | 27.874 [27.857-27.903] / 27.956 | 35.210 [35.156-35.272] / 35.233 | - |  | 2857 |
| org648 | 650 | 14856 | FlashTP-hybrid | 4.185 [4.177-4.193] / 4.254 | 4.374 [4.366-4.396] / 4.436 | - |  | 642 |
| org648 | 650 | 14856 | node-hybrid (P8) | 2.320 [2.311-2.329] / 2.376 | 2.514 [2.505-2.535] / 2.553 | 2.980 (job 1147) |  | 598 |
| org648 | 650 | 14856 | GPUStep (P9) | 2.352 [2.341-2.361] / 2.401 | 2.360 [2.347-2.371] / 2.409 | 2.409 (job 1147) | 1.07x / 1.85x / 14.92x | 597 |
| org648 | 650 | 14856 | GPUStep w/ FlashTP model | 4.437 [4.430-4.443] / 4.511 | 4.445 [4.440-4.454] / 4.516 | - | 0.57x / 0.98x / 7.92x | 639 |
| water648 | 648 | 26550 | stock | 36.389 [36.361-36.450] / 36.349 | 45.454 [45.391-45.541] / 45.312 | 45.863 (job 1084); VV+stock 44.36 |  | 4307 |
| water648 | 648 | 26550 | FlashTP-hybrid | 6.481 [6.455-6.488] / 6.494 | 6.682 [6.674-6.698] / 6.678 | - |  | 1039 |
| water648 | 648 | 26550 | node-hybrid (P8) | 2.989 [2.982-2.997] / 3.029 | 3.183 [3.173-3.201] / 3.217 | 3.643 (job 1147) |  | 981 |
| water648 | 648 | 26550 | GPUStep (P9) | 3.028 [3.020-3.036] / 3.077 | 3.044 [3.035-3.051] / 3.087 | 3.082 (job 1147) | 1.05x / 2.20x / 14.93x | 976 |
| water648 | 648 | 26550 | GPUStep w/ FlashTP model | 6.835 [6.829-6.844] / 6.887 | 6.843 [6.835-6.848] / 6.893 | - | 0.47x / 0.98x / 6.64x | 1036 |
| wbox648 | 648 | 35724 | stock | 43.214 [43.185-43.257] / 43.108 | 56.287 [56.188-56.461] / 55.912 | 55.956 (job 1129); VV+stock 54.62 |  | 5445 |
| wbox648 | 648 | 35724 | FlashTP-hybrid | 8.382 [8.377-8.388] / 8.383 | 9.584 [9.563-9.617] / 9.504 | - |  | 1339 |
| wbox648 | 648 | 35724 | node-hybrid (P8) | 3.667 [3.661-3.676] / 3.703 | 4.851 [4.830-4.883] / 4.826 | 5.236 (job 1147) |  | 1270 |
| wbox648 | 648 | 35724 | GPUStep (P9) | 3.723 [3.714-3.731] / 3.758 | 3.733 [3.721-3.742] / 3.771 | 3.773 (job 1147) | 1.30x / 2.57x / 15.08x | 1265 |
| wbox648 | 648 | 35724 | GPUStep w/ FlashTP model | 8.906 [8.902-8.912] / 8.914 | 8.914 [8.907-8.920] / 8.923 | - | 0.54x / 1.08x / 6.31x | 1333 |
| water2187 | 2187 | 99438 | stock | 120.896 [120.837-120.971] / 121.183 | 141.798 [141.535-146.744] / 141.550 | 143.919 (job 1084); VV+stock 140.58 |  | 15720 |
| water2187 | 2187 | 99438 | FlashTP-hybrid | 22.657 [22.651-22.665] / 22.703 | 23.062 [23.046-23.098] / 23.087 | - |  | 3653 |
| water2187 | 2187 | 99438 | node-hybrid (P8) | 9.181 [9.165-9.196] / 9.318 | 9.568 [9.552-9.586] / 9.699 | 10.148 (job 1147) |  | 3450 |
| water2187 | 2187 | 99438 | GPUStep (P9) | 9.248 [9.238-9.262] / 9.424 | 9.251 [9.239-9.263] / 9.425 | 9.310 (job 1147) | 1.03x / 2.49x / 15.33x | 3440 |
| water2187 | 2187 | 99438 | GPUStep w/ FlashTP model | 23.695 [23.684-23.710] / 23.955 | 23.703 [23.689-23.719] / 23.962 | - | 0.40x / 0.97x / 5.98x | 3636 |
| wbox2187 | 2187 | 120506 | stock | 136.419 [136.364-136.476] / 136.982 | 208.732 [205.967-211.363] / 207.518 | 210.120 (job 1129); VV+stock 206.12 |  | 18332 |
| wbox2187 | 2187 | 120506 | FlashTP-hybrid | 26.829 [26.819-26.850] / 27.128 | 34.515 [34.488-34.537] / 34.810 | - |  | 4295 |
| wbox2187 | 2187 | 120506 | node-hybrid (P8) | 10.659 [10.646-10.672] / 10.813 | 18.357 [18.334-18.393] / 18.480 | 18.878 (job 1147) |  | 4130 |
| wbox2187 | 2187 | 120506 | GPUStep (P9) | 10.738 [10.721-10.754] / 10.915 | 10.740 [10.728-10.755] / 10.925 | 10.839 (job 1147) | 1.71x / 3.21x / 19.43x | 4111 |
| wbox2187 | 2187 | 120506 | GPUStep w/ FlashTP model | 28.244 [28.146-28.262] / 28.544 | 28.131 [28.119-28.152] / 28.549 | - | 0.65x / 1.23x / 7.42x | 4337 |
| water5184 | 5184 | 247652 | stock | 290.492 [290.347-290.651] / 290.752 | 400.459 [397.236-409.698] / 420.390 | 397.596 (job 1084); VV+stock 392.60 |  | 38714 |
| water5184 | 5184 | 247652 | FlashTP-hybrid | 55.476 [55.456-55.506] / 55.842 | 56.811 [56.776-56.840] / 57.214 | - |  | 8905 |
| water5184 | 5184 | 247652 | node-hybrid (P8) | 21.217 [21.198-21.273] / 21.582 | 22.530 [22.502-22.558] / 22.887 | 23.265 (job 1147) |  | 8408 |
| water5184 | 5184 | 247652 | GPUStep (P9) | 21.353 [21.341-21.373] / 21.716 | 21.366 [21.348-21.415] / 21.774 | 21.544 (job 1147) | 1.05x / 2.66x / 18.74x | 8382 |
| water5184 | 5184 | 247652 | GPUStep w/ FlashTP model | 58.022 [58.003-58.048] / 58.395 | 57.914 [57.884-58.033] / 58.398 | - | 0.39x / 0.98x / 6.91x | 8863 |
| wbox5184 | 5184 | 285460 | stock | 319.315 [319.041-319.593] / 319.382 | 445.570 [387.098-470.976] / 447.309 | 437.770 (job 1084); VV+stock 432.13 |  | 43406 |
| wbox5184 | 5184 | 285460 | FlashTP-hybrid | 63.260 [63.234-63.288] / 63.616 | 105.568 [105.504-105.659] / 105.931 | - |  | 10169 |
| wbox5184 | 5184 | 285460 | node-hybrid (P8) | 23.783 [23.728-23.980] / 24.111 | 66.045 [66.010-66.078] / 66.370 | 66.844 (job 1147) |  | 9628 |
| wbox5184 | 5184 | 285460 | GPUStep (P9) | 23.903 [23.874-23.954] / 24.287 | 23.947 [23.897-23.997] / 24.312 | 24.091 (job 1147) | 2.76x / 4.41x / 18.61x | 9575 |
| wbox5184 | 5184 | 285460 | GPUStep w/ FlashTP model | 66.267 [66.242-66.293] / 66.635 | 66.279 [66.257-66.327] / 66.648 | - | 1.00x / 1.59x / 6.72x | 10122 |

**Summary (B2, GPUStep vs node-hybrid Phase 8 / FlashTP-hybrid / stock):**
* Periodic boxes, where the O(N^2) NL is gone:
  * wbox5184: 23.95 ms vs 66.0 / 105.6 / 446 (**2.76x / 4.41x / 18.6x**)
  * wbox2187: 10.74 vs 18.36 (1.71x)
  * wbox648: 1.30x
* Clusters: 1.03-1.09x vs node-hybrid, from the removed in-model sort, nonzero sync and copy-in.
* aspirin: 0.814 ms, the fastest of all backends (FlashTP-hybrid 0.910).
* wbox192 uses the proto fallback: equal to node-hybrid (0.99x).
* B3: GPUStep is 2.77x faster than Phase 8's MD at wbox5184 (24.09 vs 66.84 ms/step) and 1.53x at aspirin. Atoms*steps/s is 2.0-2.4e5 at >= 648 atoms.
  * Sync audit: 0 synchronising calls per step; 0.1/step from check().
  * B3 - B2 = 0.02-0.15 ms, so the integration is negligible.
* Capture cost: GPUStep first step (NL construction + warmup + capture) is 0.24-0.84 s.
* Memory: peak memory is equal to node-hybrid (wbox5184: 9.6 GB vs 43.4 GB stock eager). nl9 buffers are about 68 B per E_cap slot.
* Stress (stress=True) has the same graph structure and is tested; it was not timed.

## Profile after the change (job 1145, results/int9/prof_final.json, results/int9/prof_tables.md): where time moved
GPU kernel time per step (us) by category. In GPUStep, "NL_build" is only nl9's kernels; the padding/CSR rebuild shows up under elementwise/scatter (about +40-60 us).
| system | path | step ms (synced, no profiler) | GPU kernel sum us | NL_build | conv_bwd(node) | gemm(radialMLP+linears) | elementwise | fused_node | conv_fwd(node) | cueq_SC | csr_sort/search | scatter/gather/index | fused_embed | scan/compact | reduce | memcpy/memset | other | host sync API calls/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wbox5184 | node-hybrid + proto | 66.176 | 65760 | 41995 | 6441 | 5981 | 5483 | 2370 | 2294 | 466 | 182 | 231 | 232 | 0 | 20 | 61 | 5 | {'cudaMemcpyAsync': 37.0, 'cudaStreamSynchronize': 29.0} |
| wbox5184 | GPUStep | 23.976 | 23864 | 52 | 6518 | 5975 | 5546 | 2367 | 2269 | 487 | 85 | 261 | 232 | 3 | 20 | 28 | 22 | {'cudaMemcpyAsync': 1.0} |
| water5184 | node-hybrid + proto | 22.673 | 22473 | 1247 | 5585 | 5233 | 4875 | 2365 | 2032 | 468 | 176 | 207 | 204 | 0 | 19 | 58 | 5 | {'cudaMemcpyAsync': 7.0, 'cudaStreamSynchronize': 1.0} |
| water5184 | GPUStep | 21.416 | 21320 | 49 | 5651 | 5229 | 4928 | 2365 | 2020 | 486 | 83 | 237 | 202 | 3 | 19 | 29 | 20 | {'cudaMemcpyAsync': 1.0} |
| wbox2187 | node-hybrid + proto | 18.619 | 17958 | 7331 | 2851 | 2636 | 2439 | 1021 | 1012 | 220 | 142 | 140 | 106 | 0 | 15 | 41 | 6 | {'cudaMemcpyAsync': 14.0, 'cudaStreamSynchronize': 7.0} |
| wbox2187 | GPUStep | 10.778 | 10685 | 40 | 2901 | 2639 | 2484 | 1022 | 1005 | 215 | 55 | 167 | 106 | 3 | 15 | 19 | 14 | {'cudaMemcpyAsync': 1.0} |
| water2187 | node-hybrid + proto | 9.672 | 9438 | 286 | 2340 | 2207 | 2109 | 1022 | 844 | 222 | 134 | 128 | 86 | 0 | 14 | 41 | 5 | {'cudaMemcpyAsync': 7.0, 'cudaStreamSynchronize': 1.0} |
| water2187 | GPUStep | 9.294 | 9185 | 35 | 2374 | 2204 | 2146 | 1023 | 839 | 219 | 54 | 153 | 89 | 3 | 14 | 19 | 13 | {'cudaMemcpyAsync': 1.0} |
| wbox648 | node-hybrid + proto | 5.007 | 4305 | 722 | 925 | 884 | 721 | 314 | 341 | 81 | 124 | 103 | 37 | 0 | 14 | 34 | 3 | {'cudaMemcpyAsync': 13.0, 'cudaStreamSynchronize': 3.0} |
| wbox648 | GPUStep | 3.744 | 3617 | 41 | 939 | 881 | 769 | 314 | 341 | 79 | 45 | 126 | 37 | 3 | 14 | 18 | 8 | {'cudaMemcpyAsync': 1.0} |
| aspirin | node-hybrid + proto | 1.048 | 746 | 22 | 93 | 110 | 227 | 109 | 21 | 25 | 49 | 46 | 18 | 0 | 11 | 15 | 1 | {'cudaMemcpyAsync': 7.0, 'cudaStreamSynchronize': 1.0} |
| aspirin | GPUStep | 0.880 | 756 | 15 | 99 | 108 | 267 | 105 | 21 | 24 | 9 | 61 | 18 | 3 | 10 | 9 | 7 | {'cudaMemcpyAsync': 1.0} |
* **NL: 41995 -> 52 us at wbox5184** (7331 -> 40 at wbox2187, 1247 -> 49 at water5184).
* **Host syncs per step: 29-37 sync/memcpy API calls -> 0.** The single cudaMemcpyAsync left is the D2D position copy, which is non-blocking.
* **In-model CSR sort: 182 -> 85 us.** The remaining time is MACE's index-backward radix sort, which exists in all backends.
* What remains is the model itself. At 5184 atoms: conv bwd 27%, radial-MLP GEMMs 25%, elementwise 23%, fused node 10%, conv fwd 9.5%, cueq SC 2%.
  * The GPU is busy for about 99% of the step outside the profiler.
  * Next targets: the node-conv backward and the radial MLP (+ its silu/elementwise, ~45% together). The padded E_cap (20% margin) inflates all edge work by up to 20%.

## Open issues
* Small periodic boxes (L < ~15 A, fewer than 3 cell-list bins per dim) still use the proto path. That path has syncs and is not whole-step graphed. A capturable small-box NL would remove them; the brute stub is capturable but slow (4 ms).
* GPUStep B1 is slightly above node-hybrid B1 (+0.04-0.2 ms at >= 2187 atoms). The causes are the padding/CSR rebuild kernels (+~50 us) and dst-sorted edge order in the src-walking conv bwd (+~1-2%). nl9 could emit the spread padding directly.
* A cell change (NPT) is a new key and therefore a recapture. Variable cell is not supported per step.
* Graphed7's fallback runner capture does not have the gc guard that GPUStep now has. It is only relevant in processes that drop many graph objects.
* Superseded data, kept for the record:
  * `final_run1/run2` (gpu_step before the overflow-sanitisation and gc fixes; run1 wbox192 with the brute stub).
  * `final_run3/4` (wbox192/aspirin repeat).
  * `quick_gs.json` (stale NFS code).
  * `prof_base.json` gpustep:brute rows (last-node padding).

## Commands (from slurm-login-0, R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study)
```
source $R/env.sh; $R/venv-cueq/bin/python -I $R/scripts/int9/gen_systems.py                      # systems (CPU)
srun -p h100 --gres=gpu:1 -c 8 -t 00:50:00 bash $R/scripts/int9/run_test_full.sh                  # correctness (1144)
srun -p h100 --gres=gpu:1 -c 8 -t 00:15:00 bash $R/scripts/int9/job.sh test9.py $R/results/int9/test_binovf.json auto - binonly
srun -p h100 --gres=gpu:1 -c 8 -t 01:00:00 bash $R/scripts/int9/run_final.sh A prof               # timing A + profiles (TIMING_LOCK)
srun -p h100 --gres=gpu:1 -c 8 -t 01:00:00 bash $R/scripts/int9/run_final.sh B                    # timing B
TIMING=1 srun -p h100 --gres=gpu:1 -c 8 -t 00:40:00 bash $R/scripts/int9/job.sh ../md9/b3bench.py gpustep,graphed \
   aspirin,water192,wbox192,org648,water648,wbox648,water2187,wbox2187,water5184,wbox5184 --out $R/results/int9/b3_final_gpustep.json
python3 $R/scripts/int9/fin9.py $R/results/int9/final_A.json $R/results/int9/final_B.json > $R/results/int9/final_tables.md
python3 $R/scripts/int9/prof_table.py $R/results/int9/prof_final.json > $R/results/int9/prof_tables.md
# baseline (Part 1): bench9.py base_b12.json stock,hybrid+embed+node,nodehybrid+embed+node <10 systems>; prof9.py prof_base.json proto,...
```
