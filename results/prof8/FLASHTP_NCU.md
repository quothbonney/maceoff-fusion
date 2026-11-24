# Phase 8 / Agent D: Nsight Compute profile of the FlashTP-graphcap FP64 conv kernels (MACE-OFF23 medium)

Job **1043** (slurm-h100-206-077, H100 80GB HBM3 GPU-4b202516-7dca-26da-e0bf-7f5aef5ca845). **Perf counters are accessible (no ERR_NVGPUCTRPERM)**;
`ncu --set full` (39 passes/kernel) ran on every kernel. Scripts: `scripts/prof8/flash_drv.py` (capture | run | time), `job_ncu.sh`,
`parse_ncu.py`, `ncu_tables.py`. Data: `results/prof8/flashtp_full.ncu-rep` (open in Nsight Compute GUI; includes source counters),
`flashtp_full_raw.csv`, `flashtp_full_details.txt`, `flashtp_ncu_summary.json`, `flash_kernel_times.json`, `flashtp_floors.json`.
Inputs: real conv inputs (x after linear_up, y = SH, w = radial MLP output, edge_index) captured with forward-pre-hooks on
`interactions.{0,1}.conv_tp` from MACE-OFF23 medium fp64 on aspirin, water192, org192, water648, org648: `results/prof8/inputs_<sys>.pt`
(dict layer -> [x, y, w, edge_index]; reusable by B/C). The FlashTP kernels are called through the graphcap adapter (`hybrid_gc` imported first).
The kernels are `fwd_sptp_lienar_kernel_v2_shared_exp<double>` and `bwd_sptp_lienar_kernel_v1_shared_exp<double>` (sptp_exp_opt_large); fwd is
1 launch per layer, bwd is 1 launch per layer. Forces need no double backward, so bwd_bwd never runs.
Each kernel was profiled on its 2nd call; the 1st call was a warmup.

**Clock caveat.** ncu locks the SM clock to base (1.59 GHz), so ncu durations are about 1.2x the boost-clock (1.98 GHz) times. The
"t_us boost" column is the torch.profiler median of 20 unprofiled calls in the same job, at 1.98 GHz. Use it as the reference time. All the % metrics
are clock-independent ratios.

## Verdict: limiter per kernel
**Every FlashTP kernel, at every size, is latency-bound: low occupancy plus serial shared-memory dependency chains.** No throughput unit is saturated.
* **Occupancy is capped at 1 CTA per SM by shared memory.** fwd uses 116 KB of dynamic smem per CTA, 21 warps (one warp = one edge x 32 channels), so 32.8% theoretical
  and 30-31% achieved occupancy. bwd uses 228.5 KB (the sm_90 maximum is 227 KB + 1 KB reserved), 17 warps, so 26.6% / 25.6%. Registers would allow 3 fwd / 2 bwd CTAs.
  The smem is per-warp scratch of size 32 x max_ir_dim x {3 fwd, 5 bwd}, plus in2.
* **Issue slots are 39-53% used.** About 9.4 warp-cycles pass per issued instruction. The top stall reasons, from pc sampling, are about 26% each: **wait** (fixed-latency
  dependency), **short_scoreboard** (shared-memory load-to-use), and **long_scoreboard** (global gathers of x[src] and w[e]). These come from the per-CG-fiber
  read-modify-write `uvuv[k] += in1[i]*in2[j]*cg` in shared memory, which is serial within each thread, with too few warps to hide it.
* **The nearest roof is the L1/shared LSU pipe at 60-72%.** In bwd, shared stores show a 2.2-way bank conflict (3.3M conflicts at water192 L1).
  Every other unit is far from its roof: **FP64 pipe 3-9%** (0.3-1.3 TFLOP/s achieved, against a 33.5 TFLOP/s non-tensor peak), **DRAM 2-12%**
  (190-400 GB/s compulsory traffic against 3.35 TB/s), and **L2 7-19%**. The L2 atomic-input unit is only about 2% active, so **atomics are not the limiter**, even though
  fwd issues one fp64 RED per output element per edge (E x d_out: 136M REDs at water648 L1). The L2 hit rate is about 77%; output atomics stay in L2.
* **Tail and launch effects at small N.** Grid = ceil(E/21) x n_chunks fwd (n_chunks = 4 for L0, 8 for L1) and ceil(E/17) x n_chunks bwd. **aspirin (E=302): 60 / 120 CTAs
  fwd, 72 / 144 bwd on 132 SMs, which is 0.45-1.09 waves at 1 CTA/SM.** Each CTA runs one wave of edge-warps over 20 (L1) paths, so the kernel time (9-39 us) is
  pure per-warp latency. Most SMs are idle in L0. The 192- and 648-atom systems run 5-95 waves, and their per-edge time is flat
  (L1 fwd at about 34 ns/edge, L1 bwd at about 72 ns/edge).
* **bwd costs about 2x fwd** (it is the same structure, with 5 vs 3 scratch arrays, fewer warps per SM, and gw/gy written per edge). bwd uses atomics only for gx (dL/dx[src]):
  10x fewer RED instructions than fwd.

## What a node-centric design must beat
FlashTP conv time for fwd + bwd, both layers, at boost clock. Compare it against the compulsory-traffic and FP64 floors. Compulsory DRAM traffic: fwd reads w, y, x and writes out once; bwd reads w, y, x and grad_out and writes gw, gy, gx.
The dominant term is the per-edge weights w [E, W] (W = 512 / 1280) and, in bwd, gw.
FP64 cost per edge per channel: 3 x nnz(CG) + d_out. nnz/u = 16 (L0) and 86 (L1), d_out/u = 16 and 40. bwd is about 2.5x fwd.

| system | FlashTP conv fwd+bwd both layers (us) | DRAM-compulsory floor (us) | FP64 floor (us) | gap x |
|---|---|---|---|---|
| aspirin | 78.1 | 4.8 | 1.5 | 16.4 |
| water192 | 951.1 | 94.2 | 32.1 | 10.1 |
| org192 | 550.8 | 55.6 | 17.8 | 9.9 |
| water648 | 3756.5 | 372.4 | 128.5 | 10.1 |
| org648 | 2120.9 | 219.5 | 71.9 | 9.7 |

Per-kernel targets (boost us), fwd / bwd. These are the numbers B and C must beat, kernel by kernel:

| system | L0 fwd | L0 bwd | L1 fwd | L1 bwd |
|---|---|---|---|---|
| aspirin | 8.7 | 14.4 | 15.9 | 39.1 |
| water192 | 82.1 | 160.6 | 232.0 | 476.4 |
| org192 | 47.4 | 92.0 | 134.1 | 277.3 |
| water648 | 316.5 | 644.2 | 900.4 | 1895.3 |
| org648 | 178.1 | 365.6 | 504.5 | 1072.7 |

Implications for the node-centric design (B/C):
1. **FlashTP is about 10x above its own memory floor at N >= 192. The headroom comes from latency hiding, not from removing atomics.** A node-centric kernel wins only if it
   raises occupancy or ILP. Options: use less smem per warp (keep the 32 x max_ir_dim accumulators in registers, or do not stage all three/five arrays), have each warp
   process many edges per destination node with register accumulation, and break the serial shared-memory RMW chains. Removing the scatter atomics (fwd) alone is worth at most
   the L2 traffic, which is under 20% busy.
2. **The w / gw stream sets the floor.** E x W x 8 B is 272 MB at water648 L1, about 81 us each way. A dst-sorted node-centric kernel reads w through perm_dst (a gather of
   10 KB rows: fine if row-contiguous) and writes gw scattered by perm. Keep per-edge rows contiguous.
3. **At aspirin scale the kernel is a single partial wave.** A node-centric grid over N=21 nodes is even smaller. Split work across channel chunks/paths (as FlashTP does
   with gridDim.y), or across (node, chunk, path-group), to put more than 132 CTAs or warps in flight. Otherwise it will be slower than FlashTP at small N. Floors there are about 1-2 us per
   kernel, so launch latency (about 2-4 us per graph node) dominates any kernel below about 5 us.
4. **Share of end-to-end time.** In the latest graphed hybrid+embed+node replay (validate7, job 1041), FlashTP fwd+bwd is about 78 us of 744 us at aspirin (about 10%),
   951 of 2203 at water192 (about 43%), 551 of 1690 at org192 (about 33%), 3756 of 6498 at water648 (about 58%), and 2121 of 4223 at org648 (about 50%).
   A node-centric conv at 2x FlashTP speed would therefore save about 5% (aspirin), 22% (water192), and 29% (water648) of replay time.

## Per-kernel Nsight Compute table
| system | kernel | N/E | t_us (boost, torch.profiler) | t_us (ncu, 1.59 GHz lock) | grid x block | waves | regs / dyn smem | occ theor/ach % | issue % | FP64 pipe % | DRAM % | L2 % | L1/shared % | RED insts (warp) | top stalls (pc-sample %) | achieved TFLOP/s | achieved GB/s (compulsory) | floor DRAM / FP64 us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspirin | L0 fwd | 21/302 | 8.7 | 13.7 | (15, 4, 1)=60 x (32, 21, 1) | 0.45 | 32 / 115.8 KB | 32.8 / 30.6 | 39 | 3.0 | 3.6 | 6.7 | 40 | 19k | short_scoreboard 27, wait 21, long_scoreboard 18 | 0.29 | 190 | 0.5 / 0.1 |
| aspirin | L0 bwd | 21/302 | 14.4 | 22.6 | (18, 4, 1)=72 x (32, 17, 1) | 0.55 | 40 / 228.5 KB | 26.6 / 25.7 | 34 | 5.7 | 2.2 | 3.0 | 50 | 1k | long_scoreboard 25, wait 20, short_scoreboard 15 | 0.43 | 204 | 0.9 / 0.2 |
| aspirin | L1 fwd | 21/302 | 15.9 | 23.1 | (15, 8, 1)=120 x (32, 21, 1) | 0.91 | 32 / 116.1 KB | 32.8 / 30.3 | 43 | 4.4 | 5.3 | 9.0 | 50 | 48k | short_scoreboard 27, wait 22, long_scoreboard 18 | 0.72 | 256 | 1.2 / 0.3 |
| aspirin | L1 bwd | 21/302 | 39.1 | 54.7 | (18, 8, 1)=144 x (32, 17, 1) | 1.09 | 40 / 228.5 KB | 26.6 / 25.6 | 38 | 7.3 | 2.3 | 3.0 | 61 | 5k | short_scoreboard 24, wait 23, long_scoreboard 20 | 0.74 | 187 | 2.2 / 0.9 |
| water192 | L0 fwd | 192/6634 | 82.1 | 99.6 | (316, 4, 1)=1264 x (32, 21, 1) | 9.58 | 32 / 115.8 KB | 32.8 / 30.4 | 46 | 3.6 | 9.8 | 17.3 | 49 | 425k | wait 26, long_scoreboard 24, short_scoreboard 21 | 0.66 | 383 | 9.4 / 1.6 |
| water192 | L0 bwd | 192/6634 | 160.6 | 198.0 | (391, 4, 1)=1564 x (32, 17, 1) | 11.85 | 40 / 228.5 KB | 26.6 / 25.5 | 42 | 7.0 | 7.3 | 8.3 | 63 | 27k | wait 26, long_scoreboard 24, short_scoreboard 19 | 0.85 | 371 | 17.8 / 4.1 |
| water192 | L1 fwd | 192/6634 | 232.0 | 277.6 | (316, 8, 1)=2528 x (32, 21, 1) | 19.15 | 32 / 116.1 KB | 32.8 / 30.8 | 52 | 5.3 | 9.1 | 15.5 | 61 | 1061k | wait 26, short_scoreboard 26, long_scoreboard 20 | 1.09 | 334 | 23.1 / 7.6 |
| water192 | L1 bwd | 192/6634 | 476.4 | 579.1 | (391, 8, 1)=3128 x (32, 17, 1) | 23.70 | 40 / 228.5 KB | 26.6 / 25.6 | 44 | 8.6 | 7.3 | 7.5 | 72 | 106k | wait 27, short_scoreboard 26, long_scoreboard 19 | 1.33 | 309 | 43.9 / 18.9 |
| org192 | L0 fwd | 196/3670 | 47.4 | 61.8 | (175, 4, 1)=700 x (32, 21, 1) | 5.30 | 32 / 115.8 KB | 32.8 / 30.3 | 45 | 3.5 | 9.2 | 15.6 | 47 | 235k | wait 26, long_scoreboard 25, short_scoreboard 21 | 0.63 | 400 | 5.7 / 0.9 |
| org192 | L0 bwd | 196/3670 | 92.0 | 118.4 | (216, 4, 1)=864 x (32, 17, 1) | 6.55 | 40 / 228.5 KB | 26.6 / 25.7 | 41 | 6.9 | 5.8 | 7.4 | 61 | 15k | wait 25, long_scoreboard 25, short_scoreboard 18 | 0.82 | 376 | 10.3 / 2.2 |
| org192 | L1 fwd | 196/3670 | 134.1 | 161.2 | (175, 8, 1)=1400 x (32, 21, 1) | 10.61 | 32 / 116.1 KB | 32.8 / 30.8 | 52 | 5.3 | 9.5 | 14.8 | 60 | 587k | wait 26, short_scoreboard 26, long_scoreboard 20 | 1.04 | 350 | 14.0 / 4.2 |
| org192 | L1 bwd | 196/3670 | 277.3 | 337.2 | (216, 8, 1)=1728 x (32, 17, 1) | 13.09 | 40 / 228.5 KB | 26.6 / 25.7 | 43 | 8.5 | 6.9 | 6.9 | 71 | 59k | wait 26, short_scoreboard 26, long_scoreboard 19 | 1.26 | 309 | 25.6 / 10.4 |
| water648 | L0 fwd | 648/26550 | 316.5 | 369.4 | (1265, 4, 1)=5060 x (32, 21, 1) | 38.33 | 32 / 115.8 KB | 32.8 / 30.8 | 48 | 3.7 | 11.2 | 18.8 | 51 | 1699k | wait 27, long_scoreboard 26, short_scoreboard 21 | 0.69 | 391 | 36.9 / 6.5 |
| water648 | L0 bwd | 648/26550 | 644.2 | 772.7 | (1562, 4, 1)=6248 x (32, 17, 1) | 47.33 | 40 / 228.5 KB | 26.6 / 25.5 | 42 | 7.1 | 9.4 | 9.0 | 63 | 106k | wait 26, long_scoreboard 26, short_scoreboard 18 | 0.84 | 367 | 70.6 / 16.2 |
| water648 | L1 fwd | 648/26550 | 900.4 | 1067.7 | (1265, 8, 1)=10120 x (32, 21, 1) | 76.67 | 32 / 116.1 KB | 32.8 / 31.0 | 53 | 5.4 | 9.7 | 16.1 | 62 | 4248k | wait 27, short_scoreboard 25, long_scoreboard 21 | 1.12 | 338 | 90.9 / 30.2 |
| water648 | L1 bwd | 648/26550 | 1895.3 | 2294.8 | (1562, 8, 1)=12496 x (32, 17, 1) | 94.67 | 40 / 228.5 KB | 26.6 / 25.6 | 44 | 8.6 | 8.1 | 7.7 | 72 | 425k | wait 27, short_scoreboard 26, long_scoreboard 19 | 1.34 | 307 | 173.9 / 75.6 |
| org648 | L0 fwd | 650/14856 | 178.1 | 209.3 | (708, 4, 1)=2832 x (32, 21, 1) | 21.45 | 32 / 115.8 KB | 32.8 / 30.8 | 48 | 3.7 | 11.7 | 18.5 | 50 | 951k | wait 27, long_scoreboard 25, short_scoreboard 21 | 0.68 | 417 | 22.1 / 3.6 |
| org648 | L0 bwd | 650/14856 | 365.6 | 440.1 | (874, 4, 1)=3496 x (32, 17, 1) | 26.48 | 40 / 228.5 KB | 26.6 / 25.6 | 42 | 7.1 | 9.2 | 8.6 | 63 | 59k | long_scoreboard 27, wait 26, short_scoreboard 18 | 0.83 | 376 | 41.1 / 9.1 |
| org648 | L1 fwd | 650/14856 | 504.5 | 599.7 | (708, 8, 1)=5664 x (32, 21, 1) | 42.91 | 32 / 116.1 KB | 32.8 / 31.0 | 53 | 5.4 | 10.4 | 16.1 | 62 | 2377k | wait 26, short_scoreboard 26, long_scoreboard 20 | 1.12 | 364 | 54.8 / 16.9 |
| org648 | L1 bwd | 650/14856 | 1072.7 | 1298.5 | (874, 8, 1)=6992 x (32, 17, 1) | 52.97 | 40 / 228.5 KB | 26.6 / 25.6 | 43 | 8.5 | 8.3 | 7.6 | 72 | 238k | wait 27, short_scoreboard 26, long_scoreboard 20 | 1.32 | 317 | 101.5 / 42.3 |
