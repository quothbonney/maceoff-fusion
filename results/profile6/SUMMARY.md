# MACE-OFF23 medium (fp64): model-compute profile on a fixed graph (profile6)

Scripts: `scripts/profile6/prof6.py` (measure), `run6.sh` (srun wrapper), `summarize6.py` (tables → this file + `profile.json`). Raw: `raw_run1.json`; log `logs/profile6/run1.log`.
Job 1017, H100 80GB HBM3 GPU-884d19b4-3b1e-5246-0d07-a1a68b778800 (SM 1980/1980 MHz during run). Unprofiled walls agree with wl_v2_run1 to within about 10% (water648 fwd 14.2/9.9/8.7 ms vs the earlier 13.8/9.5/8.1; bwd 23.2/11.9/8.6 vs 22.9/11.6/7.7).

## Bottleneck breakdown (short)

| backend | aspirin (21 atoms) | 192 atoms | 648 atoms |
|---|---|---|---|
| e3nn | CPU-bound: GPU busy 0.9/1.4 ms in a 6.1/7.8 ms fwd/bwd wall, 84-86% idle | mixed: 21-48% idle | GPU-bound: 7-14% idle; conv (fwd 5.7, bwd 13.6 ms) and e3nn SymmetricContraction (5.7 / 6.6 ms) take about 85% of GPU time |
| flash | CPU-bound: wall about 5.8 ms flat, 83-87% idle | CPU-bound: 40-57% idle | GPU-bound: 11-22% idle; e3nn SC dominates (fwd 5.7 of 8.1 ms busy, bwd 6.6 of 10.6). FlashTP conv: 1.26 fwd / 2.6 bwd ms |
| hybrid | CPU-bound: 92-94% idle | CPU-bound: 82-91% idle | **still CPU-bound**: fwd 8.7 ms wall vs 2.5 ms GPU busy (71% idle); bwd 8.6 vs 4.1 (52% idle). GPU time is FlashTP fwd+bwd kernels (3.8 of 6.6 ms in fwd_bwd) |

- **Why hybrid is flat at about 8.7 ms fwd.** The host CPU sets the floor. The cueq SymmetricContraction costs about 4.6 ms of profiled CPU per fwd, almost all of it in `cuequivariance::uniform_1d` host-side calls (2.3 ms for P0, 0.76 ms for P1), while launching only 15 kernels that run for 37-70 us. In backward it costs another 3.6 ms of CPU in evaluate_function for 5 launches. The e3nn SC costs 1.9 ms fwd and 1.0-1.6 ms bwd of CPU. That gap of roughly 3 ms per direction is why hybrid is slower than flash at small and medium sizes, even though it has the fewest kernels (about 220 fwd and 335 bwd).
- **The CPU floor is mostly Python and dispatcher overhead, not the launch API.** Each fwd makes 250-300 launches and spends about 0.8-1.1 ms in cudaLaunchKernel/cuLaunchKernel. That is only about 10-15% of the CPU wall. Each backend's fwd needs about 9-12 ms of profiled CPU, or about 5.8-8.7 ms unprofiled. Under the CPU-bound floor the costliest ops are, per fwd: e3nn conv (1.7-1.9 ms CPU, from gather, the codegen'd TP and scatter), SC, both e3nn Linear calls (`linear` about 1.1 ms, skip_tp about 0.9 ms), spherical_harmonics (0.6 ms) and prepare_graph (0.5 ms).
- **No blocking host syncs in fwd or bwd for any backend** with check_indices=False: no cudaStreamSynchronize, cudaDeviceSynchronize, D2H memcpy, .item() or nonzero. FlashTP issues a synchronous-API `cudaMemcpyToSymbol` on each call: 9 per conv layer in fwd and 20 in bwd. All are DtoD copies, which per the CUDA docs do not block the host. Together they measure about 0.1 ms per phase. They are the only synchronous-API calls.
- **Kernel mix.** At water648 the e3nn backend's GPU time is mostly generic ATen elementwise kernels from the e3nn conv on [E, 2048/5120] fp64 tensors (BinaryFunctor mul 17%, add 14%, copy 9%), plus FP64 GEMMs (14%). With flash, the remaining GEMM, copy and elementwise time is in e3nn SC and Linear, and the FlashTP bwd/fwd kernels are 13.6% and 6.5%. With hybrid, the FlashTP kernels are 57% of GPU time.
- **Memory saved for backward.** At water648 it is 4.24 GB for e3nn, 1.74 GB for flash and 0.76 GB for hybrid. e3nn conv saves 2.9 GB of edge tensors ([26550, 5120] fp64 is 1.09 GB). The e3nn SC saves 0.99 GB, versus 0.02 GB for cueq. skip_tp saves 0.26 GB under every backend.
- **Graph construction is separate and not counted above.** Stock make_batch_dict takes 2.6 ms (aspirin) to 9.1 ms (water648). Of that, 0.5-5.1 ms is CPU work (config plus AtomicData with the neighbour list), and 2.0-4.0 ms is the per-tensor collate and transfer: 26-37 pageable H2D copies, each followed by cudaStreamSynchronize, plus 37 DtoD clone/cast copies, and no kernels. This agrees with results/graph (water648 graph_total 8.2-8.3 ms there).

Caveats:
- The profiler inflates CPU-bound walls by about 1.3-1.6x (pwall vs wall). Per-op CPU numbers are profiled values, so use them as proportions.
- One e3nn aspirin bwd profiled iteration set came out anomalously high (18.8 ms pwall vs 7.8 ms unprofiled).
- Backward ops are attributed by matching autograd sequence numbers to the forward annotation. A small remainder is unmapped (JIT DifferentiableGraphBackward and AccumulateGrad, about 17 us of GPU time).
- I did not use Nsight Systems: an nsys binary exists in the toolkit, but I did not run it. torch.profiler (CUPTI) gave sufficient kernel and API timelines.


SLURM job 1017 on slurm-h100-206-077; GPU: `0, NVIDIA H100 80GB HBM3, GPU-884d19b4-3b1e-5246-0d07-a1a68b778800, 345 MHz, 1980 MHz, 2619 MHz, 28, 69.71 W` (end: `0, NVIDIA H100 80GB HBM3, GPU-884d19b4-3b1e-5246-0d07-a1a68b778800, 1980 MHz, 1980 MHz, 2619 MHz, 32, 123.44 W`). torch 2.11.0+cu128, fp64, FlashTP check_indices=False. Graph built once per system (stock make_batch_dict), kept on GPU. Unprofiled timings: median of 30 synced reps (no hooks). Profiled: torch.profiler (CPU+CUDA, record_shapes, no stack), 3 iterations/phase with module annotations; per-iteration median. Traces: `results/profile6/trace_<backend>_<system>.json.gz`.

## 1. Phase totals
wall = unprofiled synced wall (ms); ev = CUDA-event ms; pwall = profiled wall; ksum = sum of kernel durations; busy = union of kernel+memcpy+memset intervals; idle = 1 - busy/wall (unprofiled wall); nK = kernels; launchAPI = CPU time in cudaLaunchKernel/cuLaunchKernel/ExC (profiled); mc/ms = memcpy/memset count; sync = implicit host syncs (count, ms).

| backend | system | phase | wall | ev | pwall | ksum | busy | idle | nK | launchAPI ms | mc/ms | sync (n, ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3nn | aspirin (21a/302e) | fwd | 6.11 | 6.08 | 9.84 | 0.86 | 0.87 | 0.86 | 295 | 1.02 | 3/1 | 0, 0.00 |
| e3nn | aspirin (21a/302e) | bwd | 7.78 | 7.75 | 18.78 | 1.37 | 1.38 | 0.82 | 534 | 2.47 | 8/0 | 0, 0.00 |
| e3nn | aspirin (21a/302e) | fwd_bwd | 13.76 | 13.73 | 29.83 | 2.24 | 2.25 | 0.84 | 830 | 3.58 | 11/1 | 0, 0.00 |
| flash | aspirin (21a/302e) | fwd | 5.81 | 5.78 | 8.89 | 0.73 | 0.75 | 0.87 | 252 | 0.87 | 21/1 | 18, 0.11 |
| flash | aspirin (21a/302e) | bwd | 5.67 | 5.64 | 8.36 | 0.95 | 0.98 | 0.83 | 363 | 1.11 | 23/0 | 20, 0.10 |
| flash | aspirin (21a/302e) | fwd_bwd | 11.46 | 11.43 | 17.10 | 1.68 | 1.73 | 0.85 | 616 | 2.00 | 44/1 | 38, 0.20 |
| hybrid | aspirin (21a/302e) | fwd | 8.66 | 8.63 | 12.12 | 0.46 | 0.49 | 0.94 | 221 | 0.81 | 20/0 | 18, 0.10 |
| hybrid | aspirin (21a/302e) | bwd | 8.41 | 8.38 | 10.98 | 0.65 | 0.68 | 0.92 | 334 | 1.04 | 23/0 | 20, 0.10 |
| hybrid | aspirin (21a/302e) | fwd_bwd | 17.01 | 16.98 | 23.02 | 1.12 | 1.17 | 0.93 | 556 | 1.87 | 43/0 | 38, 0.20 |
| e3nn | org192 (196a/3670e) | fwd | 5.98 | 5.95 | 9.83 | 3.13 | 3.14 | 0.48 | 295 | 1.02 | 3/0 | 0, 0.00 |
| e3nn | org192 (196a/3670e) | bwd | 7.81 | 7.78 | 11.92 | 5.02 | 5.08 | 0.35 | 539 | 1.64 | 8/10 | 0, 0.00 |
| e3nn | org192 (196a/3670e) | fwd_bwd | 13.73 | 13.70 | 21.69 | 8.16 | 8.22 | 0.40 | 835 | 2.67 | 11/10 | 0, 0.00 |
| flash | org192 (196a/3670e) | fwd | 5.90 | 5.88 | 9.35 | 2.48 | 2.51 | 0.57 | 252 | 0.90 | 21/0 | 18, 0.10 |
| flash | org192 (196a/3670e) | bwd | 5.84 | 5.80 | 8.67 | 3.22 | 3.26 | 0.44 | 366 | 1.14 | 23/7 | 20, 0.11 |
| flash | org192 (196a/3670e) | fwd_bwd | 11.78 | 11.75 | 18.16 | 5.70 | 5.77 | 0.51 | 619 | 2.05 | 44/7 | 38, 0.20 |
| hybrid | org192 (196a/3670e) | fwd | 8.66 | 8.64 | 12.00 | 0.77 | 0.80 | 0.91 | 221 | 0.81 | 20/0 | 18, 0.10 |
| hybrid | org192 (196a/3670e) | bwd | 8.49 | 8.46 | 11.02 | 1.17 | 1.21 | 0.86 | 337 | 1.07 | 23/7 | 20, 0.11 |
| hybrid | org192 (196a/3670e) | fwd_bwd | 17.19 | 17.16 | 22.93 | 1.95 | 2.01 | 0.88 | 559 | 1.87 | 43/7 | 38, 0.20 |
| e3nn | water192 (192a/6634e) | fwd | 6.24 | 6.21 | 9.82 | 3.78 | 3.78 | 0.39 | 294 | 1.03 | 3/0 | 0, 0.00 |
| e3nn | water192 (192a/6634e) | bwd | 8.33 | 8.30 | 12.24 | 6.45 | 6.55 | 0.21 | 538 | 1.63 | 8/9 | 0, 0.00 |
| e3nn | water192 (192a/6634e) | fwd_bwd | 14.33 | 14.31 | 22.21 | 10.24 | 10.35 | 0.28 | 833 | 2.66 | 11/9 | 0, 0.00 |
| flash | water192 (192a/6634e) | fwd | 5.92 | 5.90 | 9.15 | 2.62 | 2.65 | 0.55 | 252 | 0.90 | 21/0 | 18, 0.10 |
| flash | water192 (192a/6634e) | bwd | 5.86 | 5.83 | 8.63 | 3.51 | 3.55 | 0.40 | 366 | 1.14 | 23/6 | 20, 0.10 |
| flash | water192 (192a/6634e) | fwd_bwd | 11.78 | 11.75 | 17.72 | 6.14 | 6.20 | 0.47 | 619 | 2.05 | 44/6 | 38, 0.20 |
| hybrid | water192 (192a/6634e) | fwd | 8.77 | 8.74 | 12.14 | 0.93 | 0.96 | 0.89 | 221 | 0.80 | 20/0 | 18, 0.10 |
| hybrid | water192 (192a/6634e) | bwd | 8.61 | 8.57 | 11.23 | 1.49 | 1.53 | 0.82 | 337 | 1.09 | 23/6 | 20, 0.11 |
| hybrid | water192 (192a/6634e) | fwd_bwd | 17.46 | 17.43 | 23.39 | 2.43 | 2.49 | 0.86 | 559 | 1.89 | 43/6 | 38, 0.20 |
| e3nn | org648 (650a/14856e) | fwd | 11.65 | 11.63 | 13.27 | 10.00 | 10.00 | 0.14 | 300 | 1.06 | 3/0 | 0, 0.00 |
| e3nn | org648 (650a/14856e) | bwd | 17.24 | 17.21 | 17.81 | 15.47 | 15.69 | 0.09 | 555 | 1.69 | 8/12 | 0, 0.00 |
| e3nn | org648 (650a/14856e) | fwd_bwd | 28.09 | 28.06 | 29.83 | 25.47 | 25.69 | 0.09 | 856 | 2.74 | 11/12 | 0, 0.00 |
| flash | org648 (650a/14856e) | fwd | 9.48 | 9.46 | 10.91 | 7.41 | 7.43 | 0.22 | 256 | 0.91 | 21/0 | 18, 0.09 |
| flash | org648 (650a/14856e) | bwd | 10.62 | 10.59 | 11.12 | 9.30 | 9.34 | 0.12 | 380 | 1.18 | 23/8 | 20, 0.10 |
| flash | org648 (650a/14856e) | fwd_bwd | 19.38 | 19.36 | 20.90 | 16.72 | 16.79 | 0.13 | 637 | 2.09 | 44/8 | 38, 0.20 |
| hybrid | org648 (650a/14856e) | fwd | 8.78 | 8.75 | 12.28 | 1.75 | 1.77 | 0.80 | 219 | 0.82 | 20/0 | 18, 0.10 |
| hybrid | org648 (650a/14856e) | bwd | 8.63 | 8.60 | 11.29 | 2.78 | 2.83 | 0.67 | 339 | 1.10 | 23/8 | 20, 0.10 |
| hybrid | org648 (650a/14856e) | fwd_bwd | 17.36 | 17.33 | 23.25 | 4.54 | 4.61 | 0.73 | 559 | 1.88 | 43/8 | 38, 0.20 |
| e3nn | water648 (648a/26550e) | fwd | 14.17 | 14.15 | 15.43 | 12.60 | 12.60 | 0.11 | 304 | 1.06 | 3/0 | 0, 0.00 |
| e3nn | water648 (648a/26550e) | bwd | 23.19 | 23.17 | 23.81 | 21.31 | 21.67 | 0.07 | 559 | 1.71 | 8/12 | 0, 0.00 |
| e3nn | water648 (648a/26550e) | fwd_bwd | 36.63 | 36.61 | 38.07 | 33.91 | 34.28 | 0.06 | 864 | 2.76 | 11/12 | 0, 0.00 |
| flash | water648 (648a/26550e) | fwd | 9.93 | 9.91 | 11.38 | 8.09 | 8.12 | 0.18 | 256 | 0.91 | 21/0 | 18, 0.10 |
| flash | water648 (648a/26550e) | bwd | 11.88 | 11.85 | 12.33 | 10.59 | 10.63 | 0.11 | 380 | 1.18 | 23/8 | 20, 0.10 |
| flash | water648 (648a/26550e) | fwd_bwd | 21.12 | 21.10 | 22.53 | 18.69 | 18.76 | 0.11 | 637 | 2.12 | 44/8 | 38, 0.20 |
| hybrid | water648 (648a/26550e) | fwd | 8.71 | 8.68 | 12.20 | 2.46 | 2.49 | 0.71 | 219 | 0.82 | 20/0 | 18, 0.10 |
| hybrid | water648 (648a/26550e) | bwd | 8.57 | 8.53 | 11.24 | 4.09 | 4.13 | 0.52 | 339 | 1.08 | 23/8 | 20, 0.10 |
| hybrid | water648 (648a/26550e) | fwd_bwd | 17.35 | 17.32 | 23.36 | 6.56 | 6.63 | 0.62 | 559 | 1.89 | 43/8 | 38, 0.21 |

### Host-sync call sites (per call, fwd and bwd phases; explicit end-of-phase torch.cuda.synchronize excluded)

- **e3nn fwd**: none
- **e3nn bwd**: none
- **flash fwd**: 9x cudaMemcpyToSymbol in `interactions.0.conv_tp` / `GeneratedBackwardFor_flashtp_large_kernel_sptp_linear_fwd_v2`; 9x cudaMemcpyToSymbol in `interactions.1.conv_tp` / `GeneratedBackwardFor_flashtp_large_kernel_sptp_linear_fwd_v2`
- **flash bwd**: 20x cudaMemcpyToSymbol in `-` / `flashtp_large_kernel::sptp_linear_bwd_v2_shared_exp`
- **hybrid fwd**: 9x cudaMemcpyToSymbol in `interactions.0.conv_tp` / `GeneratedBackwardFor_flashtp_large_kernel_sptp_linear_fwd_v2`; 9x cudaMemcpyToSymbol in `interactions.1.conv_tp` / `GeneratedBackwardFor_flashtp_large_kernel_sptp_linear_fwd_v2`
- **hybrid bwd**: 20x cudaMemcpyToSymbol in `-` / `flashtp_large_kernel::sptp_linear_bwd_v2_shared_exp`

## 2. Time by MACE operation: aspirin (layers merged L*/P*; per-layer in profile.json)
GPU = sum of kernel time (us) launched inside the op's annotation (fwd) or by autograd nodes whose sequence number maps to a forward op inside the annotation (bwd); CPU = exclusive annotation time (fwd) / evaluate_function time (bwd), profiled (inflated ~1.0-1.6x by the profiler). Phase = fwd for fwd columns, bwd phase for bwd columns.


**e3nn**

| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |
|---|---|---|---|---|---|---|
| L*.conv | 172 | 1712 | 51 | 501 | 5050 | 178 |
| P*.symmetric_contractions | 297 | 1892 | 46 | 305 | 1614 | 34 |
| L*.linear(+/avg,reshape) | 74 | 1004 | 33 | 105 | 1887 | 62 |
| radial_embedding | 53 | 377 | 29 | 89 | 776 | 46 |
| L*.skip_tp | 60 | 845 | 23 | 76 | 1324 | 39 |
| L*.conv_tp_weights | 68 | 562 | 34 | 61 | 642 | 26 |
| spherical_harmonics | 35 | 554 | 24 | 89 | 1054 | 58 |
| prepare_graph/e0 | 26 | 581 | 12 | 49 | 474 | 26 |
| energy_sum/outputs | 19 | 356 | 9 | 31 | 521 | 21 |
| P*.linear(+sc) | 21 | 387 | 10 | 23 | 525 | 13 |
| L*.linear_up | 19 | 379 | 9 | 25 | 471 | 14 |
| readouts | 17 | 346 | 8 | 13 | 460 | 8 |
| bwd_unmapped | 0 | 0 | 0 | 10 | 245 | 7 |
| scale_shift | 6 | 84 | 4 | 1 | 37 | 1 |
| node_embedding | 4 | 171 | 2 | 0 | 0 | 0 |
| pre_interaction_glue | 1 | 45 | 1 | 0 | 0 | 0 |
| L*.interaction_other | 0 | 133 | 0 | 0 | 0 | 0 |
| P*.product_other | 0 | 75 | 0 | 0 | 0 | 0 |
| glue_after_spherical_harmonics | 0 | 19 | 0 | 0 | 0 | 0 |
| glue_after_node_embedding | 0 | 19 | 0 | 0 | 0 | 0 |

**flash**

| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |
|---|---|---|---|---|---|---|
| P*.symmetric_contractions | 295 | 1822 | 46 | 306 | 1001 | 34 |
| L*.linear(+/avg,reshape) | 74 | 1060 | 33 | 105 | 1138 | 62 |
| L*.conv | 55 | 878 | 8 | 94 | 465 | 7 |
| radial_embedding | 53 | 378 | 29 | 92 | 494 | 46 |
| L*.skip_tp | 61 | 938 | 23 | 76 | 817 | 39 |
| L*.conv_tp_weights | 68 | 542 | 34 | 62 | 418 | 26 |
| spherical_harmonics | 35 | 526 | 24 | 89 | 675 | 58 |
| prepare_graph/e0 | 25 | 497 | 12 | 50 | 297 | 26 |
| energy_sum/outputs | 20 | 352 | 9 | 32 | 319 | 21 |
| L*.linear_up | 19 | 360 | 9 | 26 | 308 | 14 |
| P*.linear(+sc) | 21 | 394 | 10 | 23 | 310 | 13 |
| readouts | 16 | 343 | 8 | 13 | 323 | 8 |
| bwd_unmapped | 0 | 0 | 0 | 10 | 166 | 7 |
| scale_shift | 7 | 83 | 4 | 1 | 21 | 1 |
| node_embedding | 4 | 213 | 2 | 0 | 0 | 0 |
| pre_interaction_glue | 1 | 43 | 1 | 0 | 0 | 0 |
| L*.interaction_other | 0 | 123 | 0 | 0 | 0 | 0 |
| P*.product_other | 0 | 72 | 0 | 0 | 0 | 0 |
| glue_after_spherical_harmonics | 0 | 16 | 0 | 0 | 0 | 0 |
| glue_after_node_embedding | 0 | 16 | 0 | 0 | 0 | 0 |

**hybrid**

| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |
|---|---|---|---|---|---|---|
| L*.linear(+/avg,reshape) | 73 | 1196 | 33 | 103 | 1241 | 62 |
| L*.conv | 53 | 936 | 8 | 90 | 462 | 7 |
| radial_embedding | 53 | 379 | 29 | 89 | 519 | 46 |
| L*.skip_tp | 60 | 948 | 23 | 72 | 949 | 39 |
| L*.conv_tp_weights | 68 | 573 | 34 | 59 | 431 | 26 |
| spherical_harmonics | 35 | 576 | 24 | 90 | 668 | 58 |
| prepare_graph/e0 | 24 | 504 | 12 | 48 | 304 | 26 |
| P*.symmetric_contractions | 37 | 4629 | 15 | 24 | 3567 | 5 |
| energy_sum/outputs | 18 | 372 | 9 | 31 | 298 | 21 |
| P*.linear(+sc) | 20 | 492 | 10 | 23 | 332 | 13 |
| L*.linear_up | 18 | 379 | 9 | 24 | 307 | 14 |
| readouts | 15 | 382 | 8 | 13 | 316 | 8 |
| bwd_unmapped | 0 | 0 | 0 | 10 | 161 | 7 |
| scale_shift | 7 | 87 | 4 | 1 | 18 | 1 |
| node_embedding | 4 | 132 | 2 | 0 | 0 | 0 |
| pre_interaction_glue | 1 | 43 | 1 | 0 | 0 | 0 |
| L*.interaction_other | 0 | 128 | 0 | 0 | 0 | 0 |
| P*.product_other | 0 | 70 | 0 | 0 | 0 | 0 |
| glue_after_spherical_harmonics | 0 | 18 | 0 | 0 | 0 | 0 |
| glue_after_node_embedding | 0 | 15 | 0 | 0 | 0 | 0 |

## 2. Time by MACE operation: water648 (layers merged L*/P*; per-layer in profile.json)
GPU = sum of kernel time (us) launched inside the op's annotation (fwd) or by autograd nodes whose sequence number maps to a forward op inside the annotation (bwd); CPU = exclusive annotation time (fwd) / evaluate_function time (bwd), profiled (inflated ~1.0-1.6x by the profiler). Phase = fwd for fwd columns, bwd phase for bwd columns.


**e3nn**

| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |
|---|---|---|---|---|---|---|
| L*.conv | 5745 | 1864 | 56 | 13644 | 3254 | 186 |
| P*.symmetric_contractions | 5709 | 1967 | 52 | 6555 | 1063 | 46 |
| L*.conv_tp_weights | 414 | 568 | 34 | 356 | 399 | 25 |
| L*.skip_tp | 321 | 875 | 21 | 276 | 859 | 39 |
| L*.linear(+/avg,reshape) | 161 | 1057 | 33 | 354 | 1176 | 62 |
| radial_embedding | 62 | 374 | 29 | 111 | 482 | 46 |
| spherical_harmonics | 45 | 635 | 24 | 107 | 822 | 58 |
| prepare_graph/e0 | 33 | 617 | 12 | 120 | 350 | 32 |
| energy_sum/outputs | 32 | 368 | 9 | 42 | 358 | 21 |
| L*.linear_up | 24 | 362 | 9 | 40 | 289 | 14 |
| P*.linear(+sc) | 27 | 403 | 10 | 30 | 358 | 13 |
| readouts | 16 | 349 | 8 | 15 | 317 | 8 |
| bwd_unmapped | 0 | 0 | 0 | 17 | 156 | 7 |
| scale_shift | 9 | 85 | 4 | 2 | 19 | 1 |
| node_embedding | 4 | 136 | 2 | 0 | 0 | 0 |
| pre_interaction_glue | 1 | 42 | 1 | 0 | 0 | 0 |
| L*.interaction_other | 0 | 127 | 0 | 0 | 0 | 0 |
| P*.product_other | 0 | 69 | 0 | 0 | 0 | 0 |
| glue_after_spherical_harmonics | 0 | 19 | 0 | 0 | 0 | 0 |
| glue_after_node_embedding | 0 | 18 | 0 | 0 | 0 | 0 |

**flash**

| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |
|---|---|---|---|---|---|---|
| P*.symmetric_contractions | 5714 | 1889 | 52 | 6553 | 1068 | 46 |
| L*.conv | 1262 | 879 | 8 | 2605 | 477 | 7 |
| L*.conv_tp_weights | 415 | 578 | 34 | 359 | 437 | 25 |
| L*.skip_tp | 324 | 890 | 21 | 276 | 829 | 39 |
| L*.linear(+/avg,reshape) | 162 | 1124 | 33 | 354 | 1162 | 62 |
| radial_embedding | 63 | 368 | 29 | 111 | 516 | 46 |
| prepare_graph/e0 | 32 | 574 | 12 | 121 | 359 | 32 |
| spherical_harmonics | 43 | 654 | 24 | 105 | 768 | 58 |
| energy_sum/outputs | 30 | 368 | 9 | 42 | 284 | 21 |
| L*.linear_up | 24 | 396 | 9 | 40 | 303 | 14 |
| P*.linear(+sc) | 27 | 404 | 10 | 30 | 348 | 13 |
| readouts | 16 | 378 | 8 | 15 | 331 | 8 |
| bwd_unmapped | 0 | 0 | 0 | 17 | 157 | 7 |
| scale_shift | 9 | 91 | 4 | 2 | 19 | 1 |
| node_embedding | 4 | 141 | 2 | 0 | 0 | 0 |
| pre_interaction_glue | 1 | 44 | 1 | 0 | 0 | 0 |
| L*.interaction_other | 0 | 123 | 0 | 0 | 0 | 0 |
| P*.product_other | 0 | 71 | 0 | 0 | 0 | 0 |
| glue_after_spherical_harmonics | 0 | 19 | 0 | 0 | 0 | 0 |
| glue_after_node_embedding | 0 | 15 | 0 | 0 | 0 | 0 |

**hybrid**

| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |
|---|---|---|---|---|---|---|
| L*.conv | 1265 | 883 | 8 | 2604 | 476 | 7 |
| L*.conv_tp_weights | 414 | 570 | 34 | 357 | 430 | 25 |
| L*.skip_tp | 322 | 920 | 21 | 272 | 950 | 39 |
| L*.linear(+/avg,reshape) | 162 | 1177 | 33 | 351 | 1246 | 62 |
| radial_embedding | 62 | 367 | 29 | 111 | 506 | 46 |
| prepare_graph/e0 | 33 | 525 | 12 | 121 | 366 | 32 |
| spherical_harmonics | 44 | 648 | 24 | 107 | 797 | 58 |
| P*.symmetric_contractions | 70 | 4684 | 15 | 64 | 3578 | 5 |
| energy_sum/outputs | 31 | 376 | 9 | 42 | 312 | 21 |
| L*.linear_up | 24 | 384 | 9 | 40 | 313 | 14 |
| P*.linear(+sc) | 28 | 513 | 10 | 30 | 334 | 13 |
| readouts | 16 | 390 | 8 | 15 | 321 | 8 |
| bwd_unmapped | 0 | 0 | 0 | 17 | 162 | 7 |
| scale_shift | 9 | 89 | 4 | 2 | 20 | 1 |
| node_embedding | 4 | 127 | 2 | 0 | 0 | 0 |
| pre_interaction_glue | 1 | 42 | 1 | 0 | 0 | 0 |
| L*.interaction_other | 0 | 135 | 0 | 0 | 0 | 0 |
| P*.product_other | 0 | 74 | 0 | 0 | 0 | 0 |
| glue_after_spherical_harmonics | 0 | 20 | 0 | 0 | 0 | 0 |
| glue_after_node_embedding | 0 | 14 | 0 | 0 | 0 | 0 |

## 3. Intermediate tensors and saved-for-backward


**aspirin: module outputs (e3nn backend; flash/hybrid identical shapes except where noted in profile.json)**

| module | outputs (shape dtype MB) |
|---|---|
| node_embedding | [21, 128] float64 0.02 |
| spherical_harmonics | [302, 16] float64 0.04 |
| radial_embedding | [302, 8] float64 0.02 |
| interactions.0.linear_up | [21, 128] float64 0.02 |
| interactions.0.conv_tp_weights | [302, 512] float64 1.24 |
| interactions.0.conv_tp | [302, 2048] float64 4.95 |
| interactions.0.linear | [21, 2048] float64 0.34 |
| interactions.0.skip_tp | [21, 2048] float64 0.34 |
| interactions.0 | [21, 128, 16] float64 0.34 |
| products.0.symmetric_contractions | [21, 512] float64 0.09 |
| products.0.linear | [21, 512] float64 0.09 |
| products.0 | [21, 512] float64 0.09 |
| interactions.1.skip_tp | [21, 128] float64 0.02 |
| interactions.1.linear_up | [21, 512] float64 0.09 |
| interactions.1.conv_tp_weights | [302, 1280] float64 3.09 |
| interactions.1.conv_tp | [302, 5120] float64 12.37 |
| interactions.1.linear | [21, 2048] float64 0.34 |
| interactions.1 | [21, 128, 16] float64 0.34; [21, 128] float64 0.02 |
| products.1.symmetric_contractions | [21, 128] float64 0.02 |
| products.1.linear | [21, 128] float64 0.02 |
| products.1 | [21, 128] float64 0.02 |
| readouts.0 | [21, 1] float64 0.00 |
| readouts.1 | [21, 1] float64 0.00 |
| scale_shift | [21] float64 0.00 |

**aspirin: tensors saved for backward (unique CUDA storages, MB; count = pack calls)**

| op | e3nn | flash | hybrid |
|---|---|---|---|
| P*.symmetric_contractions | 38.54 (18) | 38.54 (18) | 1.14 (9) |
| L*.conv | 32.91 (57) | 4.48 (32) | 4.48 (32) |
| L*.skip_tp | 10.88 (7) | 10.88 (7) | 10.88 (7) |
| L*.conv_tp_weights | 1.98 (14) | 1.98 (14) | 1.98 (14) |
| L*.linear(+/avg,reshape) | 1.84 (10) | 1.84 (10) | 1.84 (10) |
| P*.linear(+sc) | 0.39 (3) | 0.39 (3) | 0.39 (3) |
| L*.linear_up | 0.26 (2) | 0.26 (2) | 0.26 (2) |
| radial_embedding | 0.07 (21) | 0.07 (21) | 0.07 (21) |
| spherical_harmonics | 0.04 (33) | 0.04 (33) | 0.04 (33) |
| readouts | 0.02 (4) | 0.02 (4) | 0.02 (4) |
| prepare_graph/e0 | 0.01 (4) | 0.01 (4) | 0.01 (4) |
| energy_sum/outputs | 0.00 (5) | 0.00 (5) | 0.00 (5) |
| scale_shift | 0.00 (1) | 0.00 (1) | 0.00 (1) |
| **total MB** | 87.0 | 58.5 | 21.1 |

**water648: module outputs (e3nn backend; flash/hybrid identical shapes except where noted in profile.json)**

| module | outputs (shape dtype MB) |
|---|---|
| node_embedding | [648, 128] float64 0.66 |
| spherical_harmonics | [26550, 16] float64 3.40 |
| radial_embedding | [26550, 8] float64 1.70 |
| interactions.0.linear_up | [648, 128] float64 0.66 |
| interactions.0.conv_tp_weights | [26550, 512] float64 108.75 |
| interactions.0.conv_tp | [26550, 2048] float64 435.00 |
| interactions.0.linear | [648, 2048] float64 10.62 |
| interactions.0.skip_tp | [648, 2048] float64 10.62 |
| interactions.0 | [648, 128, 16] float64 10.62 |
| products.0.symmetric_contractions | [648, 512] float64 2.65 |
| products.0.linear | [648, 512] float64 2.65 |
| products.0 | [648, 512] float64 2.65 |
| interactions.1.skip_tp | [648, 128] float64 0.66 |
| interactions.1.linear_up | [648, 512] float64 2.65 |
| interactions.1.conv_tp_weights | [26550, 1280] float64 271.87 |
| interactions.1.conv_tp | [26550, 5120] float64 1087.49 |
| interactions.1.linear | [648, 2048] float64 10.62 |
| interactions.1 | [648, 128, 16] float64 10.62; [648, 128] float64 0.66 |
| products.1.symmetric_contractions | [648, 128] float64 0.66 |
| products.1.linear | [648, 128] float64 0.66 |
| products.1 | [648, 128] float64 0.66 |
| readouts.0 | [648, 1] float64 0.01 |
| readouts.1 | [648, 1] float64 0.01 |
| scale_shift | [648] float64 0.01 |

**water648: tensors saved for backward (unique CUDA storages, MB; count = pack calls)**

| op | e3nn | flash | hybrid |
|---|---|---|---|
| L*.conv | 2893.32 (57) | 387.76 (32) | 387.76 (32) |
| P*.symmetric_contractions | 994.55 (18) | 994.55 (18) | 21.69 (9) |
| L*.skip_tp | 257.53 (7) | 257.53 (7) | 257.53 (7) |
| L*.conv_tp_weights | 82.62 (14) | 82.62 (14) | 82.62 (14) |
| radial_embedding | 6.19 (21) | 6.19 (21) | 6.19 (21) |
| spherical_harmonics | 3.82 (33) | 3.82 (33) | 3.82 (33) |
| L*.linear(+/avg,reshape) | 1.84 (10) | 1.84 (10) | 1.84 (10) |
| prepare_graph/e0 | 1.27 (4) | 1.27 (4) | 1.27 (4) |
| P*.linear(+sc) | 0.39 (3) | 0.39 (3) | 0.39 (3) |
| L*.linear_up | 0.26 (2) | 0.26 (2) | 0.26 (2) |
| readouts | 0.10 (4) | 0.10 (4) | 0.10 (4) |
| energy_sum/outputs | 0.02 (5) | 0.02 (5) | 0.02 (5) |
| scale_shift | 0.01 (1) | 0.01 (1) | 0.01 (1) |
| **total MB** | 4241.9 | 1736.4 | 763.5 |

## 4. Stock graph construction (make_batch_dict; measured once, backend-independent)

| system | graph_cpu ms (config+AtomicData) | graph_total ms (incl. collate+H2D+fp64) | total-cpu ms | H2D/D2D memcpys | cudaStreamSynchronize | kernels |
|---|---|---|---|---|---|---|
| aspirin | 0.52 | 2.58 | 2.05 | 74 (Memcpy HtoD:37, Memcpy DtoD:37) | 37 | 0 |
| org192 | 1.18 | 3.59 | 2.41 | 72 (Memcpy HtoD:35, Memcpy DtoD:37) | 37 | 0 |
| water192 | 1.50 | 3.94 | 2.44 | 69 (Memcpy HtoD:32, Memcpy DtoD:37) | 37 | 0 |
| org648 | 3.55 | 6.97 | 3.42 | 66 (Memcpy HtoD:29, Memcpy DtoD:37) | 37 | 0 |
| water648 | 5.07 | 9.05 | 3.98 | 63 (Memcpy HtoD:26, Memcpy DtoD:37) | 37 | 0 |

## 5. Top-10 kernels (fwd_bwd phase = model(compute_force=True), one iteration)


**e3nn / aspirin** (kernel sum 2.24 ms)

| kernel | us | % | n |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 245 | 10.9 | 120 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 231 | 10.3 | 95 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> ` | 187 | 8.4 | 101 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::nat` | 128 | 5.7 | 25 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 97 | 4.3 | 31 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul> >(in` | 92 | 4.1 | 82 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<double, double, double, at::native` | 80 | 3.6 | 62 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 80 | 3.6 | 24 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize32x64x32_stage3_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 78 | 3.5 | 3 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 63 | 2.8 | 1 |

**flash / aspirin** (kernel sum 1.68 ms)

| kernel | us | % | n |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 187 | 11.2 | 87 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> ` | 128 | 7.6 | 71 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 96 | 5.7 | 31 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 82 | 4.9 | 31 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 80 | 4.7 | 24 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize32x64x32_stage3_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 78 | 4.6 | 3 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<double, double, double, at::native` | 73 | 4.4 | 57 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 63 | 3.8 | 1 |
| `void bwd_sptp_lienar_kernel_v1_shared_exp<double>(double const*, double const*, double const*, double const*, ` | 60 | 3.6 | 2 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul> >(in` | 55 | 3.3 | 54 |

**hybrid / aspirin** (kernel sum 1.12 ms)

| kernel | us | % | n |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 132 | 11.8 | 78 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> ` | 87 | 7.8 | 58 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 76 | 6.8 | 24 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_c` | 74 | 6.6 | 23 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<double, double, double, at::native` | 72 | 6.5 | 57 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul> >(in` | 57 | 5.1 | 56 |
| `void bwd_sptp_lienar_kernel_v1_shared_exp<double>(double const*, double const*, double const*, double const*, ` | 56 | 5.0 | 2 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 42 | 3.7 | 25 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 31 | 2.8 | 16 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::(anonymous namespace)::pow` | 27 | 2.4 | 6 |

**e3nn / water648** (kernel sum 33.91 ms)

| kernel | us | % | n |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 5759 | 17.0 | 95 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> ` | 4726 | 13.9 | 101 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 2922 | 8.6 | 120 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize64x128x16_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel__5x_c` | 2475 | 7.3 | 3 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x128x16_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel__5x_c` | 2331 | 6.9 | 4 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::nat` | 1492 | 4.4 | 24 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul> >(in` | 1334 | 3.9 | 82 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::Opaq` | 1091 | 3.2 | 9 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x32x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_alignc8_exec` | 910 | 2.7 | 11 |
| `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<t` | 849 | 2.5 | 4 |

**flash / water648** (kernel sum 18.69 ms)

| kernel | us | % | n |
|---|---|---|---|
| `void bwd_sptp_lienar_kernel_v1_shared_exp<double>(double const*, double const*, double const*, double const*, ` | 2541 | 13.6 | 2 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize64x128x16_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel__5x_c` | 2471 | 13.2 | 3 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x128x16_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel__5x_c` | 2333 | 12.5 | 4 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 1664 | 8.9 | 87 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 1287 | 6.9 | 31 |
| `void fwd_sptp_lienar_kernel_v2_shared_exp<double>(double const*, double const*, double const*, double*, int co` | 1220 | 6.5 | 2 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> ` | 1186 | 6.3 | 71 |
| `std::enable_if<true, void>::type internal::gemvx::kernel<int, int, double, double, double, double, true, true,` | 581 | 3.1 | 6 |
| `sm80_xmma_gemm_f64f64_f64f64_f64_nt_n_tilesize32x32x16_stage5_warpsize2x2x1_tensor8x8x4_execute_kernel__5x_cub` | 565 | 3.0 | 4 |
| `std::enable_if<true, void>::type internal::gemvx::kernel<int, int, double, double, double, double, true, true,` | 529 | 2.8 | 8 |

**hybrid / water648** (kernel sum 6.56 ms)

| kernel | us | % | n |
|---|---|---|---|
| `void bwd_sptp_lienar_kernel_v1_shared_exp<double>(double const*, double const*, double const*, double const*, ` | 2539 | 38.7 | 2 |
| `void fwd_sptp_lienar_kernel_v2_shared_exp<double>(double const*, double const*, double const*, double*, int co` | 1223 | 18.6 | 2 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 269 | 4.1 | 78 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize32x64x32_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel__5x_cu` | 238 | 3.6 | 9 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> ` | 222 | 3.4 | 58 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<double, double, double, at::native` | 192 | 2.9 | 57 |
| `sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize128x64x16_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel__5x_c` | 154 | 2.3 | 1 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul> >(in` | 146 | 2.2 | 56 |
| `void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x64_16x4_nt_align1>(cutlass_80_tensorop_d884gemm_32x64_16` | 129 | 2.0 | 3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1>(cutlass_80_tensorop_d884gemm_64x32_16` | 122 | 1.9 | 3 |
