# Phase 8 / Agent D: integrating the node-centric conv into the graphed pipeline, plus benchmarks

**"node-hybrid"** = node-centric fp64 conv + cuEq SymmetricContraction + fused_embed + fused_node, run under the graphed7 GraphedRunner.
- The conv is C's `scripts/nodetp8_bwd/nodeconv.install`: B's fwd `k_<hash>p1`, C's src-centric fused bwd `kern`, and one CSR build per model call that both layers share.
- The model is built with `hybrid_gc.build_calc(conv="e3nn", sc="cueq")`, then `NC.install`, then `FE.install`, then `FN.install`.
- Build code: `scripts/prof8/cfg8.py` (`build("nodehybrid+embed+node")`).
- The baseline is the latest validated graphed **hybrid+embed+node**: FlashTP-graphcap conv + cuEq SC + both fused ops (validate7).

Scripts: `scripts/prof8/{cfg8,gate8,time8,conv8,summ8,job,job_conv}.py/.sh`. gate8 and time8 are copies of validate7's gate7 and time7, with the build switched to cfg8. gate8 tightens the stress tolerance to the phase-8 value of 1e-14.
Results: `results/prof8/gate_nodehybrid.json`, `conv_j1065.json`, `time_whole.json` (j1066), `time_whole_j2.json` (j1067), `INTEGRATION.json`, `integ_tables.md`.
All jobs ran on slurm-h100-206-077, one H100 each, at SM 1980 MHz:

| job | purpose | GPU |
|---|---|---|
| 1064 | gate | GPU-4b202516 |
| 1065 | conv-only | GPU-884d19b4 |
| 1066 | whole model, run 1 | GPU-3ef4e875 |
| 1067 | whole model, run 2, with conv attribution | GPU-4b202516 |

## Verdict
* **Correctness: PASS, 0 failures** (job 1064). Two configs: nodehybrid+embed+node and nodehybrid without the fused ops. Both were checked against eager stock e3nn fp64.
  * Coverage is the same 8 tests as validate7. Eager fused vs stock. Graphed runs over H2O, aspirin, water192, org192, water648, org648, wbox192, wbox648, single C and a far O2 pair (the last two have 0 edges), each with a 6-step 0.02 A walk (neighbour lists change) plus 3 repeats. Stress on wbox192/648 with forced recapture. A forced recapture. Species-order and different-N traps. LRU eviction. The stock builder.
  * Worst values: |dE|/atom 1.07e-12 eV (gate 1e-11), max|dF| 6.5e-13 eV/A (aspirin reversed species, where |F| reaches about 300 eV/A; typical about 1e-14), |dsigma| 1.1e-16 eV/A^3 (gate 1e-14).
  * Conv-only local error vs e3nn on real inputs: max rel 2.3e-15 (out, gx, gy, gw).
* **Whole model: node-hybrid+embed+node is 2.2-2.3x faster per graph replay than hybrid+embed+node at 648 atoms, and 1.25-1.44x faster at 192 atoms.** It is **5-11% slower at H2O and aspirin.** Full-step speedups are 1.6-2.0x at 648 atoms and 1.2-1.3x at 192 atoms.
  * The two jobs agree to within 0.3%.
  * The conv's share of replay time drops from 54-72% to 36-38% (192/648 atoms). At small N it rises from 13% to 18-21%.
* **Why small N regresses.** At aspirin the node conv (fwd+bwd, both layers, plus the CSR build) costs 169 us of GPU time inside the replay, against FlashTP's 97 us. Three causes:
  * The CSR build is about 35 us of radix sort.
  * C's bwd uses grid=N CTAs, which is only 21 CTAs on 132 SMs at aspirin.
  * The extra kernels/memsets add launch gaps.
  The graph has 33 fewer GPU ops (255 vs 288), but that does not make up for it.
  * Fix options: drop the CSR sort when the runner can supply dst-sorted edges, and split bwd by channel chunk at small N (FLASHTP_NCU.md item 3). Alternatively, dispatch FlashTP when E < ~2000.

## Whole model, graphed (job 1067; median [p10-p90] ms; >= 10 warmups, 60 reps)
Replay = graph.replay() only. Full step = runner.step_tensors: GPU neighbour-list build + copy-in + replay, no stress, no D2H. Conv+CSR = GPU time of the FlashTP fwd/bwd kernels, or of the node fwd/bwd kernels plus the radix-sort/searchsorted kernels, measured in one profiled replay.
The conv kernels run on the padded edge count E_cap (+20%).
Caveat: about 30-50 us of radix/searchsorted kernels from elsewhere in the model also match the pattern in both configs (hybrid shows 4545 us here vs 4493 us for FlashTP kernels alone in j1066).
| system (N, E -> E_cap) | hybrid+embed+node replay / step ms | node-hybrid+embed+node replay / step ms | replay speedup | step speedup | conv+CSR GPU us in replay: hybrid -> node (fraction of replay) | GPU ops in graph | peak MiB | first step s (hybrid/node) | j1066 replay hybrid / node (repro) |
|---|---|---|---|---|---|---|---|---|---|
| H2O (3, 6->64) | 0.605 [0.604-0.605] / 1.148 [1.135-1.161] | 0.676 [0.676-0.677] / 1.281 [1.264-1.296] | 0.89x | 0.90x | 76 (13%) -> 123 (18%) | 300 -> 267 | 131 -> 67 | 13.399 / 0.664 | 0.600 / 0.669 |
| aspirin (21, 302->384) | 0.754 [0.753-0.756] / 1.381 [1.369-1.398] | 0.791 [0.789-0.792] / 1.415 [1.403-1.428] | 0.95x | 0.98x | 97 (13%) -> 169 (21%) | 288 -> 255 | 79 -> 77 | 0.063 / 0.060 | 0.754 / 0.793 |
| water192 (192, 6634->8000) | 2.205 [2.202-2.208] / 2.856 [2.846-2.873] | 1.528 [1.524-1.534] / 2.163 [2.152-2.179] | 1.44x | 1.32x | 1181 (54%) -> 551 (36%) | 297 -> 282 | 312 -> 296 | 0.088 / 0.065 | 2.208 / 1.527 |
| org192 (196, 3670->4416) | 1.691 [1.689-1.695] / 2.341 [2.328-2.351] | 1.349 [1.346-1.352] / 1.988 [1.974-2.006] | 1.25x | 1.18x | 662 (39%) -> 364 (27%) | 298 -> 283 | 214 -> 202 | 0.064 / 0.077 | 1.693 / 1.347 |
| water648 (648, 26550->31872) | 6.498 [6.493-6.509] / 7.192 [7.177-7.219] | 3.001 [2.989-3.009] / 3.669 [3.653-3.687] | 2.17x | 1.96x | 4545 (70%) -> 1097 (37%) | 301 -> 286 | 1039 -> 981 | 0.078 / 0.073 | 6.500 / 2.994 |
| org648 (650, 14856->17856) | 4.216 [4.211-4.222] / 4.905 [4.886-4.947] | 2.336 [2.329-2.343] / 3.014 [2.991-3.042] | 1.80x | 1.63x | 2558 (61%) -> 726 (31%) | 301 -> 286 | 642 -> 598 | 0.072 / 0.068 | 4.212 / 2.328 |
| wbox648 (648, 35698->42880) | 8.359 [8.353-8.367] / 10.064 [10.021-10.110] | 3.672 [3.664-3.681] / 5.354 [5.318-5.405] | 2.28x | 1.88x | 6028 (72%) -> 1394 (38%) | 301 -> 286 | 1339 -> 1271 | 0.119 / 0.083 | 8.362 / 3.670 |

Setup and capture cost:
* build+install: hybrid 5.4 s (includes the model load, the cuEq conversion and the first FlashTP build); node-hybrid 0.48 s once the B/C load_inline extensions are cached in `$TORCH_EXTENSIONS_DIR`.
* First step (graph build + warmup + capture): 0.06-0.12 s for both configs. The first H2O step is 13.3 s for hybrid (the first-ever capture in the process is JIT/Triton/cuEq warmup) and 0.64 s for node-hybrid.
* Peak memory is equal or slightly lower with node-hybrid (water648 1039 -> 981 MiB). Graph node count falls by 15-33.

## Conv-only comparison on identical real inputs (job 1065; CUDA-graph replay of fwd+bwd, ms, median of 60)
Inputs are `results/prof8/inputs_<sys>.pt`, captured from the model through FlashConv pre-hooks. grad_out is fixed random. "bwd" = autograd.grad w.r.t. (x, y, w).
* **Ours** includes a CSR build on every call (csr_cache off). This is conservative: in the model the CSR is built once for both layers.
* **[A]** columns are agent A's numbers (results/oeq8/BASELINES.md, job 1056/1051/1057). A used separately captured but identical systems from stock e3nn, graph replay timed with events, 50 reps. The method matches: A's FlashTP-gc numbers agree with mine to within 1-4%, e.g. water648 L1 2.833 vs 2.839. So A's results are reused, not re-run.
* **LunaTP "kernel"** excludes make_graph and the y/w permutation. make_graph cannot be captured (host readback).
* Eager timings, bwd-only, GPU-kernel sums, kernel counts and peak memory per impl are in `integ_tables.md` / `conv_j1065.json`.

| system/layer | E | ours node (incl. CSR build) | FlashTP-gc | e3nn | [A] FlashTP-gc | [A] cuEq fused | [A] OEQ atomic | [A] OEQ det (kernel) | [A] LunaTP (kernel; graph prebuilt, y/w pre-permuted) | ours vs FlashTP | ours vs LunaTP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aspirin/L0 | 302 | **0.109** (fwd 0.068) | 0.061 (fwd 0.031) | 0.187 | 0.057 | 0.043 | n | 0.263 | 0.034 | 0.56x | 0.31x |
| aspirin/L1 | 302 | **0.110** (fwd 0.074) | 0.095 (fwd 0.040) | 0.596 | 0.090 | 0.114 | 0.039 | 0.633 | 0.045 | 0.86x | 0.41x |
| water192/L0 | 6634 | **0.305** (fwd 0.134) | 0.293 (fwd 0.096) | 0.952 | 0.286 | 0.137 | 0.116 | 0.195 | 0.088 | 0.96x | 0.29x |
| water192/L1 | 6634 | **0.337** (fwd 0.173) | 0.763 (fwd 0.264) | 4.394 | 0.754 | 0.419 | 0.305 | 0.705 | 0.151 | 2.26x | 0.45x |
| org192/L0 | 3670 | **0.181** (fwd 0.081) | 0.184 (fwd 0.067) | 0.601 | 0.178 | 0.101 | 0.074 | 0.146 | 0.065 | 1.02x | 0.36x |
| org192/L1 | 3670 | **0.211** (fwd 0.117) | 0.462 (fwd 0.164) | 2.631 | 0.456 | 0.314 | 0.186 | 0.602 | 0.119 | 2.19x | 0.56x |
| water648/L0 | 26550 | **0.401** (fwd 0.185) | 1.009 (fwd 0.343) | 3.264 | 0.999 | 0.409 | 0.385 | 0.337 | 0.215 | 2.52x | 0.54x |
| water648/L1 | 26550 | **0.676** (fwd 0.271) | 2.839 (fwd 0.931) | 16.367 | 2.833 | 1.151 | 1.179 | 0.992 | 0.484 | 4.20x | 0.72x |
| org648/L0 | 14856 | **0.308** (fwd 0.171) | 0.595 (fwd 0.208) | 1.909 | 0.587 | 0.278 | 0.233 | 0.228 | 0.139 | 1.93x | 0.45x |
| org648/L1 | 14856 | **0.475** (fwd 0.227) | 1.626 (fwd 0.537) | 9.365 | 1.625 | 0.903 | 0.702 | 0.672 | 0.302 | 3.42x | 0.64x |
| wbox648/L0 | 35698 | **0.443** (fwd 0.204) | 1.328 (fwd 0.449) | 4.313 | 1.320 | 0.520 | 0.511 | 0.425 | 0.279 | 2.99x | 0.63x |
| wbox648/L1 | 35698 | **0.892** (fwd 0.317) | 3.780 (fwd 1.235) | 21.851 | 3.797 | 1.361 | 1.593 | 1.269 | 0.643 | 4.24x | 0.72x |

Summary:
* Ours beats FlashTP-gc by **2.5-4.2x at 648 atoms**, 2.2x on L1 at 192 atoms, and about 1.0x on L0 at 192 atoms. It is **0.56-0.86x (slower) at aspirin**: the fixed costs (CSR sort about 35-45 us, grid=N bwd) dominate there.
* At 648 atoms ours beats cuEq fused, OEQ atomic and OEQ det on both layers. At 192 atoms L0 it does not.
* **LunaTP's kernels are 1.4-3x faster than ours** (water648 L1 0.48 vs 0.68 ms; part of that gap is our CSR build). Luna's numbers exclude its graph build and y/w permutation; ours include the CSR build.
* By kernel: at water648 the node conv kernels (fwd+bwd, both layers) take 832 us of GPU time vs FlashTP's 3731 us (4.5x). The two bwd `kern` launches are 615 us of that 832, so **bwd is now the target**. The fwd kernel is 217 us.

## LunaTP whole model (task 3): not run. This is a measured bound instead.
I did not install cuequivariance into venv-luna. There was not enough time for the install, the adapter and the gate.
The bound below takes the node-hybrid replay, removes our in-model conv+CSR time, and adds A's LunaTP kernel graph f+b (both layers) scaled by E_cap/E for padding.
It is optimistic for Luna. It leaves out:
* make_graph, which needs a host sync per step and has data-dependent table sizes. It must run outside the graph and would force recapture or a bound on table sizes.
* The in-graph y/w gather into dst order: about 0.1-0.2 ms at water648 (an [E,1280] fp64 gather).

| system | node-hybrid replay ms | node conv+CSR in replay ms | LunaTP kernel graph f+b L0+L1 (A, unpadded) ms | x E_cap/E | bound: LunaTP-hybrid replay ms | vs node-hybrid |
|---|---|---|---|---|---|---|
| aspirin | 0.791 | 0.169 | 0.079 | 1.27 | 0.722 | 1.10x |
| water192 | 1.528 | 0.551 | 0.239 | 1.21 | 1.266 | 1.21x |
| org192 | 1.349 | 0.364 | 0.184 | 1.20 | 1.206 | 1.12x |
| water648 | 3.001 | 1.097 | 0.699 | 1.20 | 2.743 | 1.09x |
| org648 | 2.336 | 0.726 | 0.441 | 1.20 | 2.140 | 1.09x |
| wbox648 | 3.672 | 1.394 | 0.922 | 1.20 | 3.385 | 1.08x |

So even an ideal LunaTP drop-in would improve the node-hybrid replay by at most about 8-21%, and in practice by less once its non-capturable graph build and the y/w permutation are added. Its paper's "19% end to end over FlashTP" claim does not transfer directly to this case. **Against FlashTP-hybrid, our node-hybrid already delivers 2.2x at water648.** The remaining conv headroom is about 0.35 ms at water648 (Luna kernel bound 0.84 ms vs ours 1.10 ms in-graph).

## Gate detail
### Correctness gate (job 1064), tol dE/atom 1e-11, dF 1e-11, dsigma 1e-14

| config | test | ok | n | worst dE/atom | worst max dF | worst dsigma |
|---|---|---|---|---|---|---|
| nodehybrid+embed+node | eager_fused_vs_stock | True | 20 | 7.2e-13 | 1.4e-14 | 2.9e-17 |
| nodehybrid+embed+node | systems_walk_repeat | True | 90 | 9.0e-13 | 1.4e-14 | 0.0e+00 |
| nodehybrid+embed+node | stress_periodic_recapture | True | 17 | 1.5e-13 | 7.1e-15 | 1.1e-16 |
| nodehybrid+embed+node | forced_recapture | True | 5 | 1.5e-13 | 8.9e-15 | 0.0e+00 |
| nodehybrid+embed+node | bug_a_species_order | True | 6 | 1.7e-13 | 6.5e-13 | 0.0e+00 |
| nodehybrid+embed+node | bug_b_diffN_interleaved | True | 13 | 1.5e-13 | 5.3e-15 | 0.0e+00 |
| nodehybrid+embed+node | lru_eviction | True | 5 | 3.0e-13 | 4.2e-15 | 0.0e+00 |
| nodehybrid+embed+node | stock_builder | True | 6 | 1.5e-13 | 4.0e-15 | 2.9e-17 |
| nodehybrid | eager_fused_vs_stock | True | 20 | 9.0e-13 | 8.9e-15 | 2.4e-17 |
| nodehybrid | systems_walk_repeat | True | 90 | 1.1e-12 | 2.5e-14 | 0.0e+00 |
| nodehybrid | stress_periodic_recapture | True | 17 | 1.5e-13 | 8.9e-15 | 5.6e-17 |
| nodehybrid | forced_recapture | True | 5 | 1.5e-13 | 7.1e-15 | 0.0e+00 |
| nodehybrid | bug_a_species_order | True | 6 | 0.0e+00 | 5.7e-13 | 0.0e+00 |
| nodehybrid | bug_b_diffN_interleaved | True | 13 | 1.5e-13 | 5.3e-15 | 0.0e+00 |
| nodehybrid | lru_eviction | True | 5 | 3.0e-13 | 4.4e-15 | 0.0e+00 |
| nodehybrid | stock_builder | True | 6 | 1.5e-13 | 3.8e-15 | 1.8e-17 |

failures: []
