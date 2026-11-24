# design6: fusion boundaries beyond FlashTP for MACE-OFF23 medium (fp64, energy+forces, H100)

Scope: analysis plus two small GPU measurements. No production code was written.
- Segment microbenchmark: `scripts/design6/seg_bench.py`, SLURM job 1019, node slurm-h100-206-077, GPU-3ef4e875-4571-3cea-d408-241db662acb5. Results in `results/design6/seg_bench.json`, log in `logs/design6/seg_bench_2.log`. Job 1018 crashed because MACE-OFF23 passes `cutoff=None`; that was fixed and the run repeated.
- FlashTP capture check: `scripts/design6/cap_check.py`, job 1020, log `logs/design6/cap_check.log`.
- The bounds use my own profiler kernel sums (job 1019) plus `workloads/summary_v2.md` (jobs 1008 and 1012), `graph_e2e/e2e_run1.json` (job 1013) and `perf/prof_main.json`. They are cross-checked against `results/profile6/SUMMARY.md` (job 1017), which arrived late; see the end of section 0.

## 0. The measurement that decides everything

Model: hybrid (FlashTP conv + cueq SC), `model(bd, compute_force=True)`, median of 30 runs. Kernel sum is per call, from torch.profiler.

| system | N | E | eager wall ms | GPU kernel sum ms | kernels/call | GPU busy |
|---|---|---|---|---|---|---|
| aspirin | 21 | 302 | 15.61 | 1.22 | 634 | 8 % |
| org192 | 196 | 3670 | 17.25 | 2.09 | 644 | 12 % |
| water648 | 648 | 26550 | 17.46 | 6.98 | 645 | 40 % |

All three sizes are **host/dispatch bound**: wall time is close to flat (15.6 to 17.5 ms) while GPU work grows 6x. The flat ~10 ms of flash64 and ~16 ms of hybrid at small sizes is Python/dispatcher/launch time, not kernel time. Consequences:
- A kernel fusion helps the eager wall clock **only through the host dispatch it removes**. That is the segment's eager time minus about 0.1 to 0.15 ms for the fused op's own launch. The GPU time it removes does not help eager wall time.
- With CUDA-graph replay the wall clock drops to about the GPU time, and only then do GPU-side fusion savings count.

Below, both regimes are reported for each candidate.

### Per-segment fwd+bwd (bwd with respect to whatever carries grad in the force pass)

Columns: eager wall (includes host overhead) / CUDA-graph replay (GPU only), in ms.

| segment | aspirin | org192 | water648 | kernels |
|---|---|---|---|---|
| L0 radial MLP 8→64→64→64→512 | 0.47 / 0.072 | 0.47 / 0.109 | 0.48 / 0.335 | 31 |
| L1 radial MLP …→1280 | 0.47 / 0.076 | 0.46 / 0.135 | 0.62 / 0.482 | 32 |
| of which tail 64→W only (L0 / L1, replay) | 0.015 / 0.019 | 0.027 / 0.056 | 0.131 / 0.281 | 2 to 3 |
| L0 FlashTP conv (eager / kernel sum) | 0.54 / 0.056 | 0.56 / 0.184 | 1.20 / 1.09 | 30 |
| L1 FlashTP conv (eager / kernel sum) | 0.54 / 0.088 | 0.75 / 0.475 | 3.09 / 3.03 | 30 |
| L0 linear, /avg, skip_tp | 1.60 / 0.179 | 1.61 / 0.321 | 1.59 / 0.684 | 81 |
| L1 linear, /avg | 0.80 / 0.097 | 0.80 / 0.151 | 0.82 / 0.338 | 43 |
| L0 product (cueq SC + linear + sc) | **6.22** / 0.077 | 6.27 / 0.091 | 6.24 / 0.158 | 31 |
| L1 product | **2.12** / 0.040 | 2.13 / 0.047 | 2.13 / 0.081 | 17 |
| readouts L0 + L1 | 0.67 / 0.050 | 0.68 / 0.052 | 0.68 / 0.054 | 24 |
| embedding (sh + Bessel radial, with respect to vectors) | 1.71 / 0.300 | 1.75 / 0.328 | 1.86 / 0.388 | 169 |

At aspirin the segment replay times add up to about 1.03 ms, against a 1.22 ms model kernel sum. CUDA-graph replay adds only about 0.13 to 0.25 µs per kernel node over the kernel sum (embedding: 0.300 vs 0.278 ms for 169 kernels).

The cueq SymmetricContraction costs **8.3 ms of host time per call for 0.12 to 0.24 ms of GPU work**. That is why hybrid is flat at about 16 ms. By difference, the e3nn SC used in flash64 costs about 2.6 ms of host time (hybrid 15.85 − flash64 10.10 = 8.3 − x).

### Tensors crossing the boundaries (fp64, MB)

| tensor | aspirin (E=302) | org192 (E=3670) | water648 (E=26550) |
|---|---|---|---|
| edge_feats [E,8] / sh [E,16] | 0.02 / 0.04 | 0.23 / 0.47 | 1.7 / 3.4 |
| MLP hidden [E,64] (×3 per layer, saved for bwd) | 0.15 | 1.9 | 13.6 |
| tp_weights L0 [E,512] / L1 [E,1280] | 1.2 / 3.1 | 15 / 38 | **109 / 272** |
| FlashTP out L0 [N,2048] / L1 [N,5120] | 0.34 / 0.86 | 3.2 / 8.0 | 10.6 / 26.5 |
| FlashTP bwd dL_din2 partials [E,16·upath] (L0 / L1) | 0.15 / 0.31 | 1.9 / 3.8 | 13.6 / 27 |

At aspirin and 200 atoms, every intermediate fits in the 50 MB L2, so fusion saves almost no HBM traffic. Only at about 650 atoms do the tp_weights (and the dW of the same size in bwd) become real traffic: about 0.4 GB in fwd and about 1.2 GB in bwd, roughly 0.4 to 0.5 ms at about 2.7 TB/s.

### What the force backward needs

Energy is a scalar, so forces = −∂E/∂positions need the full reverse pass: positions → vectors → (sh, lengths → Bessel×envelope → MLP → W) → FlashTP → linear/skip → SC → readout.

Saved state (MACE 0.3.17 + FlashTP):
- FlashTP saves `in1, in2(sh), weight, src, dst` plus metadata (`fused_e3nn_exp_opt_large.py:148-165`). Its bwd kernel regathers in1[src], sh[e], W[e] and grad_out[dst].
- It writes dW [E,W] per edge without atomics, dL_din2 per (edge, chunk) (reduced afterwards, `:274`), and atomicAdds dL_din1 into src (`bwd_sptp_linear_shared_exp_double.cu:206`).
- The e3nn MLP saves its inputs and SiLU pre-activations, about 6×[E,64] per layer.
- MACE-OFF23 applies the envelope inside the radial embedding (`radial_embedding` returns `cutoff=None`, `blocks.py:436`), so the `tp_weights*cutoff` multiply at `blocks.py:914-915/1018-1019` is not executed. There is nothing to fuse there.
- Inference needs no parameter grads (the calculator freezes them), so the MLP bwd is dX only.
- Any fused kernel must either save W or h3, or recompute W from h3 in bwd.

### FlashTP kernel facts relevant to fusion (cgpatch tree, `flashTP_e3nn/sptp_exp_opt_large/`)

- **Grid** is `(ceil(E/bb), upath_cnt)`, with the block = 32 channels × bb edges (`fwd_..._double.cu:195-196`, `bwd_..._double.cu:273-274`). blockIdx.y is the (in1 irrep, 32-channel chunk). That gives 4 chunks for L0 and 8 for L1, so **no block ever sees all 128 channels of a node or all paths of an edge**.
- **Shared memory**, computed from `find_optimal_abc` (`fused_e3nn_exp_opt_large.py:737-760`) and the launch formula (`fwd:214`, `bwd:290`):
  - fwd: bb=32, 8·32·(32·21+17) B ≈ **172 KB**;
  - bwd: bb=17, 8·17·(32·52+16) B ≈ **223 KB** of the 227 KB opt-in.
  - Either way it is 1 block/SM.
  - I compute these from the code. The ~116 KB in the brief does not match this formula for lmax=3 fp64. Treat the 116 KB figure as unverified until profile6 reports `smem/block`.
- **At aspirin** the fwd grid is 10×4 (L0) or 10×8 (L1) blocks on 132 SMs, so the GPU is underfilled. At 650 atoms it is 830×8.
- **Scatter** is an `atomicAdd` into out[dst] (`fwd:155`). A node's message is final only after the whole grid finishes.
- **Per-call host work:**
  - 9× `cudaMemcpyToSymbol` of metadata into `__constant__` (`fwd:201-212`; bwd similar);
  - `std::cout << shared_memory_bytes` on every fwd call (`fwd:216`);
  - `cudaFuncSetAttribute` (`fwd:222`);
  - `torch.zeros` output (`py:66`), plus a dead `mem_debug = empty_like(grad_output)` [N,out] in bwd (`py:270`);
  - the adapter's int64→int32 `edge_index` copies (`flashtp_adapter_v2.py:247-248`) and an optional `aminmax` host sync (`:242-243`).
- **Launches** use `<<<grid, block, smem>>>` with no stream argument (`fwd:226`, `bwd:301`, `bwd_bwd:352`), so they go to the **legacy default stream**.

### Cross-check against profile6 (SUMMARY.md arrived after the above was written: job 1017, a different GPU)

| quantity | profile6 | this study (job 1019) |
|---|---|---|
| hybrid fwd_bwd GPU busy, aspirin / org192 / water648 (ms) | 1.17 / 2.01 / 6.63 | 1.22 / 2.09 / 6.98 (kernel sum) |
| hybrid fwd_bwd idle fraction, same sizes | 93 % / 88 % / 62 % | — |
| cueq SC host CPU per call (ms) | ~4.6 fwd (in `uniform_1d` host calls) + ~3.6 bwd ≈ 8.2 | 8.3 (eager segment) |
| e3nn SC host CPU per call (ms) | 1.9 fwd + 1.0-1.6 bwd ≈ 2.9-3.5 | 2.6 (inferred) |

- flash (e3nn SC) GPU busy is 1.73 / 5.77 / 18.76 ms. At 650 atoms, flash is GPU-bound on the e3nn SC, which is why replay bounds (4) use the cueq-SC (hybrid) kernel set.
- profile6 also confirms the following:
  - no blocking host syncs in fwd/bwd with `check_indices=False`;
  - FlashTP issues 9 (fwd) + 20 (bwd) `cudaMemcpyToSymbol` per layer-call, all DtoD, about 0.1 ms per phase;
  - only about 10 to 15 % of host time is in the launch API itself. The rest is Python and dispatcher, so fewer, larger ops (fusion) and replay are both effective, and replay more so.
- Graph construction: 26-37 pageable H2D copies, each followed by a sync.
- Using profile6's own numbers changes no bound by more than about 5 %.

## 1. Candidates

Notation: S = strongest existing backend:
- aspirin and ~200 atoms: FlashTP-FP64CG (flash64). Stock ASE call 13.39 / 14.46 ms (org192); with the GPU graph builder 10.25 (H2O proxy) / 10.43 (water192).
- ~650 atoms: hybrid. 24.97 stock (water648), 16.48 with the GPU builder.

Amdahl bounds remove only the measured time of the fused segment. "Eager" means today's host-bound execution. "Replay" means after (4).

### (1) Radial basis + MLP → FlashTP (generate W on the fly)

- **Boundary:** h3 [E,64] (or edge_feats [E,8]) enters instead of W [E,512 or 1280]. This removes the W write/read in fwd and the dW write/read in bwd: 109+272 MB each way at water648, about 0.4 to 0.5 ms of HBM.
- **Compute added:**
  - 2·E·64·W FLOP per pass (4.35 GFLOP for L1 at water648);
  - in-kernel, fp64, not tensor-core GEMM, in a 1-block/SM kernel, about 0.15 to 0.3 ms;
  - bwd must **recompute W** (or keep saving it, which defeats the purpose) and produce dh3 = dW·M_lastᵀ.
- **Structural problem:** the 1280 W columns of an edge are split across the 8 blockIdx.y chunks, so dh3 needs atomics on [E,64] or an extra reduction kernel. M_last is 64×1280 fp64 (655 KB), so it streams from L2, about 164 KB per chunk per edge batch.
- **Shared memory:** fwd has ~55 KB headroom, enough for a 32×64 h3 tile (16 KB). bwd has ~4 KB headroom, so bb drops from 17 to about 15. The full 3-layer MLP (8→64→64→64) inside the kernel would also need SiLU pre-activations for bwd: about 3×16 KB per tile. That does not fit in bwd, so recompute.
- **Gain:**
  - Eager: removes about 2×0.47 ms of MLP dispatch, so roughly −0.85 ms (−1.0 at 650).
  - Replay: removes ≤ 0.15 / 0.30 / 1.2 ms of GPU (MLP plus W traffic, before paying the recompute). Realistic net at 650 is about 0.5 to 0.8 ms.
- **Difficulty:** high. It changes the FlashTP fwd and bwd, adds atomics, and needs new parity tests.

### (2) FlashTP neighbour sum → linear → /avg_num_neighbors → skip_tp

- **Boundary:** out [N,2048] / [N,5120], 10.6 / 26.5 MB at 650. That is negligible traffic.
- **True in-kernel fusion is blocked.**
  - The linear mixes all 128 channels of an irrep, but FlashTP blocks hold 32-channel chunks.
  - The sum is complete only after a grid-wide atomicAdd. That needs a grid-wide sync or a dst-sorted CSR rewrite of FlashTP, and its bwd would need Lᵀ·grad applied at gather.
  - The gain over a separate kernel is only the 10 to 27 MB round trip, under 0.03 ms.
- **Cheap, exact alternative (recommended):** an algebraic fold done once at load time.
  - L0: `skip_tp(linear(m)/avg, onehot)` with one-hot `node_attrs` (10 elements) equals a per-species linear: M_{l,z} = L_l · W_skip[l,:,z,:] / avg. That is 4 l × 10 z × 128×128 fp64 = 5.2 MB of weights, and the whole thing becomes one species-gathered batched GEMM.
  - L1: fold 1/avg into the linear weights.
  - This turns 81+43 kernels into 2 to 4 kernels. It is fp64-exact up to reassociation (~1e-16 relative). Saved tensors for bwd: the message and the species index only.
- **Gain:**
  - Eager: −2.25 ms (1.60+0.80 dispatch, minus about 0.15).
  - Replay: −0.25 / −0.43 / −0.90 ms GPU.
- **Difficulty:** low (Python plus a cuBLAS batched GEMM). Risk: keeping the state_dict/key compatibility of the adapter.

### (3) Symmetric contraction → linear (+sc) → readout

- **Boundary:** node tensors [N,128·16] and smaller, at most 2.7 MB. GPU time is tiny: 0.08 to 0.24 ms for both layers.
- **Gain:**
  - Eager, with cueq SC: removes about 8.3 ms (cueq) + 0.67 (readouts) − 0.3 ≈ **8.7 ms**. With e3nn SC (flash64): about 3.0 ms.
  - Replay: ≤ 0.15 ms.
- **What the measurement shows:** the bottleneck is cueq's per-call host path (31 + 17 kernels taking 8.3 ms of host time), not kernel boundaries.
  - The segment captured into a CUDA graph without problems (replay 0.077 / 0.040 ms). The cuet JIT/plan happens at construction or first call, not per call.
  - Any of the following gets the same win: a small fused node-wise kernel, calling cuet's lower-level segmented-polynomial op directly, or capturing just this sub-graph. The last needs no FlashTP change, because this segment contains no FlashTP.
- **Difficulty:** low to medium (a partial graph needs static N, but N is fixed for MD).
- **Note:** this makes hybrid the best backend at *all* sizes.

### (4) Fixed-shape CUDA-graph capture of the existing op sequence (replay, not fusion)

What breaks capture, from the code plus job 1020:
- **FlashTP launches on the legacy stream.** Job 1020, capturing FlashTP L0 fwd+bwd at water648, failed with `cudaErrorStreamCaptureImplicit` ("operation would make the legacy stream depend on a capturing blocking stream").
- **Silent wrong-graph hazard.** In job 1019, at the end of a long process, the same capture *returned* a graph whose replay took 0.108 ms, below the 1.09 ms the FlashTP kernels alone need. So the kernels were not in the graph. Any capture of FlashTP must be checked by replay-vs-eager parity, not by "no exception".
- **FlashTP per-call host work:** `cudaMemcpyToSymbol` per call (legacy-stream copies), `cudaFuncSetAttribute`, `std::cout`.
  - Fix: `at::cuda::getCurrentCUDAStream()` in all three launches.
  - Constant-table upload once at init, or `cudaMemcpyToSymbolAsync` on the current stream.
  - Drop the print.
  - The experiment agent's `src/flashTP-graphcap` diff targets this.
- **Dynamic shapes:** E changes every MD step.
  - Pad E to a bucket, with dummy edges src=dst=a padding node and zero sh/W. Pad N by one sink node. FlashTP's atomicAdd tolerates this.
  - Recapture when a bucket overflows.
- **Graph construction:** the GPU graph prototype uses `nonzero()` (`proto_graph.py:100,121,123`, which syncs the host) and a dynamic-length `argsort`. It must stay outside the graph or become a fixed-capacity neighbour list.
- **Host syncs in the model path:** the adapter's `aminmax` and `int()` when `check_indices=True` (use False). MACE `get_outputs` and `torch.autograd.grad` are capturable (the segment graphs in job 1019 captured autograd.grad).
- **Python-side allocations** (`torch.zeros` / `empty`) are fine under the capture memory pool. The cueq SC segment captured fine.

Gain bound: wall ≈ GPU kernel sum + ~0.2 µs/node + host per-call floor. From the job-1019 kernel sums (hybrid), the model fwd+bwd goes to about 1.4 / 2.3 / 7.2 ms.
- With the stock CPU graph build: add graph_total 2.70 / 3.49 / 8.27 and about 0.6 of calculator overhead, giving about 4.7 / 6.4 / 16.1 ms.
- With the GPU builder: about 0.45 ms of non-model overhead (16.48 hybrid_proto − 16.06 fwd_bwd), giving about 1.85 / 2.75 / 7.65 ms.

Padding adds up to one bucket of edges. That is GPU time that grows linearly with E, about +5 to 10 % at 650 atoms.

### (5) Persistent kernel spanning several ops or both interaction layers

- **Dependencies force grid-wide barriers:**
  - every scatter (atomicAdd into out[dst]) must complete before the linear;
  - layer 1 gathers node_feats[src] from layer 0's *finished* product output;
  - the backward reverses all of this, plus atomicAdd into dL_din1 / dvectors.
  - That is about 10 to 12 `grid.sync()` per energy+forces call, which is cheap (µs).
- **The real blockers:**
  - FlashTP needs 172 to 223 KB smem per block, so it cannot co-reside with any other stage. A persistent grid is forced to 1 block/SM, and all other stages run at that low occupancy.
  - fp64 tensor-core GEMMs (MLP, linear) and the cueq SC would have to be re-implemented inside the kernel.
  - The whole forward plus a hand-written reverse pass would replace torch autograd. Every saved tensor needs a global-memory home (W or h3, sh, in1, SC intermediates).
  - Variable E needs either padding (as in 4) or device-side work queues.
- **What it can win beyond (4) + targeted fusions:** removal of the remaining ~640 → ~10 kernel boundaries, and latency hiding when the grid is tiny.
  - Aspirin: GPU 1.22 ms over 634 kernels, an average of 1.9 µs per kernel, so it is latency dominated. A plausible floor is a 40-stage dependency chain × ~10 µs ≈ 0.4 ms.
  - 200 atoms: about 0.8 ms.
  - 650 atoms: about 4.5 ms (FlashTP compute ~3.7 ms is irreducible here).
  - But most of the small-size gap is also removed by fusing the elementwise chains into a few kernels inside the replayed graph: embedding 169 kernels → 1, MLP 63 → 2 to 4, linear/skip 124 → 2 to 4.
  - **Increment unique to a persistent kernel: about 0.1 to 0.3 ms** (the remaining ~20 to 40 boundaries × ~3 to 5 µs).

## 2. Ranked opportunities: maximum whole energy+forces speedup vs S

Each cell is the speedup over S in the form stock CPU graph build / GPU graph builder. "e" = applied in today's eager mode; "r" = applied on top of (4) replay, versus the replay result.

| rank | boundary | type | aspirin | ~200 at | ~650 at | difficulty | main risks |
|---|---|---|---|---|---|---|---|
| 1 | (4) whole fwd+bwd CUDA-graph replay (padded E, FlashTP stream fix) | replay | **2.85× / 5.5×** | **2.3× / 3.4-3.8×** | **1.55× / 2.15×** | medium | silent mis-capture (observed); bucket recapture; padding; parity needed per bucket |
| 2 | (3) SC→linear→readout one op / cueq host-path fix | host-overhead removal (fusion or partial capture) | e: 1.29× / 1.41× · r: ≤1.03× | e: 1.26× / 1.40× | e: 1.53× / 2.1× (reaches the GPU floor) · r: ≤1.02× | low-med | cuet API stability; mul_ir/ir_mul layouts |
| 3 | (2) fold linear·/avg·skip_tp into species-indexed GEMM (separate kernel, not in FlashTP) | algebraic fold + fusion | e: 1.20× / 1.28× · r: 1.06× / 1.16× | e: 1.18× / 1.27× · r: 1.07× / 1.19× | e: 1.10× / 1.16× · r: 1.06× / 1.13× | low | fp64 reassociation (~1e-16); state_dict compatibility |
| 4 | (1) radial MLP fused into FlashTP fwd/bwd | kernel fusion | e: 1.07× / 1.09× · r: ≤1.03× / 1.09× | e: 1.06× / 1.09× · r: ≤1.05× / 1.12× | e: 1.04× / 1.07× · r: ≤1.08× / 1.19× (realistic ~1.1×) | high | bwd smem (4 KB headroom), cross-chunk dh3 atomics, recompute cost eats the gain |
| 5 | (5) persistent megakernel (both layers, fwd+bwd) | genuine persistent kernel | ≤3.6× / ≤12× vs S; **≤1.1× over rank 1-4 plus fused elementwise chains** (≤ about 0.3 ms) | ≤2.9× / ≤8× vs S; ≤1.1× over 1-4 | ≤1.9× / ≤3.3× vs S; ≤1.05× over 1-4 | very high | re-implement FlashTP, cueq SC and fp64 GEMMs; hand-written reverse pass; 1-block/SM occupancy; maintainability |

Not one of the five, but measured: fusing the embedding (sh + Bessel + envelope, 169 kernels, 1.7 ms eager / 0.30 ms GPU) into one kernel gives e: −1.6 ms (about 1.15× / 1.2× at small sizes) and r: −0.25 ms (1.06× / 1.15× at aspirin). It is low difficulty and comes with an analytic backward.

### Assumptions behind the bounds

- Wall time is taken as max(host, GPU). Eager is host bound at all three sizes (section 0).
- Eager savings = the segment's eager time − 0.1 to 0.3 ms for the replacement op.
- Small-size baselines use flash64. Its e3nn-SC host cost (~2.6 ms) is inferred, not measured.
- The ~200-atom GPU-builder baseline is water192 (E=6634). The model GPU time is from org192 (E=3670), hence the 3.4 to 3.8 range.
- Replay = kernel sum + 0.2 µs/node. The megakernel floors are estimates, not measurements.
- The bounds of the different rows do not simply multiply. Rows 2 and 3 in eager mode overlap with row 1: once replayed, their host savings are already counted.

## 3. Is a persistent megakernel justified for MACE-OFF23 medium energy+forces at 21 to 650 atoms?

**No.**

1. **Today's cost is host dispatch, not kernel boundaries on the GPU.** The GPU is 8 % / 12 % / 40 % busy. CUDA-graph replay of the existing kernels (rank 1) already removes about 85 to 90 % of the attainable time at small sizes. Aspirin goes from 10.25 to about 1.85 ms with the GPU builder, of which about 0.45 ms is a Python/calculator floor that no kernel design touches.
2. **The remaining GPU time is dominated by kernels a megakernel cannot shrink.**
   - At 650 atoms, FlashTP is 4.1 of 7.0 ms of GPU time. It is compute- and smem-bound at 1 block/SM, so a megakernel would inherit that occupancy for every stage.
   - The fusions that do move GPU time (rank 3, rank 4, embedding) can each be done as an ordinary kernel inside the replayed graph. Together they give about 1.5 to 2 ms at 650 atoms and about 0.5 to 0.7 ms at aspirin.
3. **What only a persistent kernel adds** is about 20 to 40 inter-kernel gaps of ~3 to 5 µs, plus some latency hiding on tiny grids. That is about 0.1 to 0.3 ms, at most about 1.1× after ranks 1 to 4.
4. **The cost is a hand-written fp64 forward and full reverse pass** (forces need every edge and node intermediate), re-implementing FlashTP fwd/bwd, the cueq SC and fp64 tensor-core GEMMs inside one kernel. That kernel would have to hold ~10 grid-wide barriers and atomics while staying within 227 KB smem and at 1 block/SM. Each MACE, cueq or FlashTP update would invalidate it.

Revisit only if the target becomes single-molecule MD at ≲50 atoms where a sub-1 ms step matters, *after* ranks 1 to 4 plus embedding fusion have been measured and still sit above about 1 ms of GPU time.

**Recommended order:**
1. (4) replay, once the FlashTP stream/constant-memory fix is in (experiment agent), with replay-vs-eager parity gating.
2. (3) cueq SC host path, or partial capture.
3. (2) the linear/skip fold.
4. Embedding fusion.
5. (1) only if ≥1000-atom systems matter.
