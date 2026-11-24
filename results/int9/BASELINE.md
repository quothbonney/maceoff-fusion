# int9 BASELINE (Phase 9, agent C) -- B1/B2 at scale for existing backends (job 1069) [profiler section appended below]

Systems: results/int9/systems/*.extxyz (scripts/int9/gen_systems.py, deterministic): water clusters water{192,648,2187,5184}
(flash_proto.systems.water_cluster n_side 4/6/9/12, seed 0), periodic water boxes wbox{192,648,2187,5184} (graph_profile
water_box, density 1.0, seed 1, cubic L = 12.4/18.6/27.9/37.3 A), org648 + aspirin (copied from results/workloads), H2O,
zeroC (single C), farO2 (O-O at 20 A, 0 edges).
Raw: results/int9/base_b12.json, tables results/int9/base_tables.md, log logs/int9/base_b12.log. Script scripts/int9/bench9.py.
Job 1069, slurm-h100-206-077, H100 80GB GPU-884d19b4, SM 1980 MHz; 10 warmups, 60 synced reps, median [p10-p90] ms.
Concurrent on other GPUs: md9 prep/correctness jobs (no other timing job; TIMING_LOCK held).

## Boundaries (exact)
* stock B1 = eager `model(batch, compute_force=True)` on a prebuilt stock batch (no graph). stock B2 = `calc.calculate(atoms)`
  = matscipy NL on CPU + AtomicData + H2D + eager fwd/bwd + D2H (stock has no GPU-resident path). Stock at >= 2187 atoms: 2 warm, 10 reps (call > 0.2 s).
* graphed (GraphedRunner, FlashTP hybrid+embed+node / node-hybrid+embed+node): **B1 = graph.replay()** (prebuilt NL in static buffers).
  **B2 = GPU-resident step** = `ProtoGPU.build(pos_gpu)` (scripts/int9/common9.py: proto_graph's O(N^2) neighbour_list on GPU,
  periodic wraps floor(pos@inv(cell)) on GPU, no H2D; `nonzero()` -> >= 1 blocking host sync per chunk) + `_Graph.load`
  (copy edges/shifts into static buffers + padding) + `replay()`. "NL only" = ProtoGPU.build alone.
  "host-proto" = Phase 8 full step `runner.step_tensors(atoms)` (host positions -> pinned H2D + same NL) for reference.

## B1 / B2 (ms)
| system | N | E | stock B1 / B2 | FlashTP-hybrid B1 / B2 | node-hybrid B1 / B2 | proto NL in B2 | NL share of node-hybrid B2 | peak MiB stock / hyb / node |
|---|---|---|---|---|---|---|---|---|
| aspirin | 21 | 302 | 11.28 / 15.99 | **0.676** / **0.908** | 0.767 / 0.947 | 0.09 | 10% | 112 / 143 / 77 |
| water192 | 192 | 6634 | 13.07 / 17.75 | 2.194 / 2.381 | **1.518 / 1.703** | 0.09 | 6% | 1132 / 312 / 296 |
| wbox192 | 192 | 10588 | 15.67 / 21.86 | 2.939 / 3.589 | **1.729 / 2.379** | 0.52 | 22% | 1623 / 443 / 423 |
| org648 | 650 | 14856 | 27.74 / 35.01 | 4.187 / 4.383 | **2.328 / 2.508** | 0.10 | 4% | 2857 / 642 / 597 |
| water648 | 648 | 26550 | 36.20 / 45.41 | 6.477 / 6.667 | **2.987 / 3.177** | 0.11 | 3% | 4307 / 1039 / 981 |
| wbox648 | 648 | 35724 | 43.03 / 56.14 | 8.381 / 9.512 | **3.669 / 4.799** | 1.02 | 21% | 5446 / 1339 / 1270 |
| water2187 | 2187 | 99438 | 120.7 / 141.2 | 22.55 / 22.94 | **9.165 / 9.553** | 0.32 | 3.4% | 15718 / 3653 / 3450 |
| wbox2187 | 2187 | 120506 | 136.2 / 205.5 | 26.82 / 34.50 | **10.66 / 18.35** | **7.58** | **41%** | 18330 / 4361 / 4131 |
| water5184 | 5184 | 247652 | 291.4 / 411.4 | 55.81 / 57.13 | **21.23 / 22.59** | 1.26 | 5.6% | 38714 / 8971 / 8473 |
| wbox5184 | 5184 | 285460 | 319.6 / 446.1 | 63.55 / 106.4 | **23.82 / 66.31** | **42.34** | **64%** | 43404 / 10231 / 9690 |
(E_cap = round_cap(1.2 E, 64) for the graphed backends; p10-p90 spreads are < 1% except stock B2 at 5184 (+-8%), see base_tables.md.)

Setup: build+install stock 0.25 s, FlashTP-hybrid 5.4 s, node-hybrid 0.67 s. First capture 0.06-0.5 s (FlashTP-hybrid first-ever
capture 13 s = JIT warm-up). No OOM up to 5184 atoms: stock eager fp64 peaks at 43 GB (wbox5184), graphed ~1.9 MB/atom
(9.7 GB). Rough capacity on 80 GB: stock ~9-10k atoms of water, graphed node-hybrid ~40k atoms (memory) -- but periodic B2 then
is dominated by the O(N^2 x 27 images) proto NL (scales ~N^2: 7.6 ms -> 42 ms for 2187 -> 5184).

## Verdict on the neighbour list (asked: is the O(N^2) builder the limiting factor at 2000+ atoms?)
* **Periodic boxes: YES.** The proto O(N^2) builder (27 periodic images, chunked brute force + nonzero host syncs) is 41% of the
  node-hybrid B2 step at 2187 atoms (7.6 of 18.3 ms) and **64% at 5184 atoms (42.3 of 66.3 ms)**; it costs more than the whole
  model (23.8 ms replay). It also scales as N^2 while the model scales as E ~ N. A cell-list NL (O(N), ~0.1-1 ms expected)
  would make node-hybrid B2 at wbox5184 ~24-25 ms (2.7x) and at wbox2187 ~11 ms (1.7x).
* **Non-periodic clusters: NO.** One image only: proto NL is 0.32 ms (3.4%) at 2187 and 1.26 ms (5.6%) at 5184. There the
  model replay (fwd+bwd) is 94-96% of B2; B2 - B1 overhead is 0.19-0.4 ms (NL + nonzero sync + load copies).
* **What limits clusters (and periodic once the NL is fixed): the model replay**, ~0.083-0.092 us per edge at 2187-5184
  atoms (see profile section for the split; conv bwd + radial MLP GEMMs + elementwise dominate at large E).
* Small systems (aspirin): FlashTP-hybrid is still 12% faster in B1 (0.676 vs 0.767 ms); part of node-hybrid's deficit is the
  in-model CSR sort (~35 us) that the external-CSR path removes (Phase 9 measurement in INTEGRATION.md).
* The host-proto path (Phase 8 B2) adds only 0.3-0.75 ms vs the GPU-resident proto; the H2D copy is not the problem.

## Profile of B2 at the largest sizes (node-hybrid + proto; job 1100, torch.profiler CPU+CUDA, 3 steps; scripts/int9/prof9.py)
Raw: results/int9/prof_base.json. GPU kernel time per step (us), NL kernels = kernels launched inside the ProtoGPU.build
window (ends at the nonzero() sync). (record_function annotations, which the raw JSON also lists as GPU events, are
subtracted here; prof9.py now filters them.) GEMM = cuBLAS sm90 fp64 xmma (radial MLP [E_cap, 8-64-64-64-512/1280] fwd+bwd
dominates, plus linears not covered by fused_node). Elementwise = ATen unary/silu/add/fill (radial MLP activations, gradients).

| system | step ms | NL (proto) | conv bwd (node) | GEMM | elementwise | fused_node | conv fwd (node) | cueq SC | CSR sort+search | index/scatter | fused_embed | host sync API calls / step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wbox5184 | 66.0 | **41924 (64%)** | 6418 | 5974 | 5480 | 2350 | 2291 | 464 | 182 | 232 | 232 | 29 cudaStreamSynchronize + 37 cudaMemcpyAsync (27 images -> 13 chunks x nonzero) |
| water5184 | 22.6 | 1244 (5.5%) | 5561 (25%) | 5222 (23%) | 4942 (22%) | 2355 | 2030 | 467 | 175 | 206 | 204 | 1 sync |
| wbox2187 | 18.5 | **7320 (41%)** | 2852 | 2635 | 2436 | 1015 | 1009 | 221 | 141 | 140 | 104 | 7 syncs |
| water2187 | 9.64 | 287 (3%) | 2332 (25%) | 2205 (23%) | 2110 (22%) | 1016 | 842 | 221 | 133 | 127 | 85 | 1 sync |
| wbox648 | 5.00 | 722 (17%) | 921 | 882 | 721 | 313 | 342 | 81 | 124 | 102 | 37 | 3 syncs |
GPU idle inside B2 (wall - busy) is small (0.15-0.3 ms/step; the nonzero() syncs stall the host, the GPU queue is short anyway).

**Conclusion for agents A/B:** at 2000+ atoms the O(N^2) builder is THE limiter for periodic systems (41-64% of B2, many
host syncs); for clusters the model is (conv bwd ~25%, radial-MLP GEMMs ~23%, elementwise ~22%, fused node op ~10%, conv fwd ~9%).
The in-model CSR sort is small at scale (0.13-0.18 ms, < 1.5%) but ~35 us fixed (matters at aspirin size).

## Pitfall found while integrating (affects any builder that puts padding on node N-1)
A whole-step graph with the builder's padding convention (all E_cap - E padding self-edges on the LAST node) made the
node-centric conv and MACE's index backward serial on that node: GPUStep step 126 ms (water2187) / 317 ms (water5184) vs
9.2 / 21 ms (prof_base.json gpustep:brute rows: conv bwd 89-221 ms, conv fwd 23-58 ms, index backward 7-1290 ms).
Fix in scripts/int9/gpu_step.py `_spread_padding`: padding edges re-homed to nodes k mod N with sort-free CSR rebuild
(O(E) elementwise + 2 scatters, ~tens of us); after the fix GPUStep water2187 = 9.30 ms (prof_q.json).
