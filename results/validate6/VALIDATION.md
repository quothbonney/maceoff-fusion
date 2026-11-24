# validate6: independent check of the cudagraph6 CUDA-graph prototype (MACE-OFF23 medium, fp64)

Under test: `scripts/cudagraph6/graphed_model.py` (unmodified), with FlashTP `src/flashTP-graphcap` @ 415e7a3 (graph-capture build), set up through `scripts/cudagraph6/hybrid_gc.py` in the same way as `run_cg6.py` (check_indices=False).
Jobs (all on node slurm-h100-206-077, H100 80GB, max SM clock 1980 MHz, measured at 1980 MHz at the end of each job):
* 1024: parity + checks 1-3 + timing run A, GPU-4b202516-7dca-26da-e0bf-7f5aef5ca845
* 1025: timing run B, GPU-884d19b4-3b1e-5246-0d07-a1a68b778800
* 1026: rerun of check 2b (recapture) and check 3 after I fixed a bug in my own script, same GPU as 1025

Jobs 1024 and 1025 ran at the same time on different GPUs of the same node.

Scripts: `scripts/validate6/` (common6.py, v_correct.py, v_parity.py, v_timing.py, upstream_check.py, job_*.sh). Logs: `logs/validate6/`.
Raw data: `correct.json` (check 1, check 2 MD), `correct_rerun.json` (checks 2b and 3), `parity_graphcap_vs_cgpatch.json`, `upstream_fp64check_graphcap_*.json`, `timing_A.json`, `timing_B.json`, `validation_summary.json`.

## Findings

| # | check | result | severity | evidence |
|---|---|---|---|---|
| 1 | Recorded kernels in replay | PASS | - | One profiled `graph.replay()` contains 4 FlashTP kernels in fp64: `fwd_sptp_lienar_kernel_v2_shared_exp<double>` and `bwd_sptp_lienar_kernel_v1_shared_exp<double>`, once per layer each. For hybrid it also contains 6 cueq `segmented_polynomial_*` kernels. These counts equal eager. Total GPU events: replay 599 vs eager 601 (aspirin hybrid), 661/663 (aspirin flash), 610/612 (water648 hybrid), 689/691 (water648 flash). The summed kernel time is no more than the replay time measured with CUDA events: aspirin hybrid 1.15 vs 1.36 ms, aspirin flash 1.70/1.91, water648 hybrid 7.47/7.67, water648 flash 19.6/19.7. |
| 2 | 200-step NVE on water192 (eager e3nn, 0.5 fs, 300 K, seed 1234), 41 frames | PASS | - | Edge count over the run is 6152-6634 against E_cap 8000, so no recapture happened. Graphed vs eager e3nn on each frame's exact configuration: hybrid max \|dE\|/atom 1.5e-13, max \|dF\| 5.3e-15, max RMS 1.7e-15; flash 1.5e-13 / 4.0e-15 / 1.5e-15. The GPU graph builder's input vs the stock MACECalculator forces: max \|dF\| 3.1e-15. |
| 2b | Forced recaptures: water192 compressed to scale 0.78 and back, starting from a tight cap of 6656 | PASS | - | Two recaptures per backend (6656→8704 at E=6934, 8704→11008 at E=8772). The steps that recapture agree with eager to \|dE\|/atom ≤1.5e-13 and \|dF\| ≤1.3e-14. Shrinking E back down (padding refilled) is also correct. At the most compressed geometry (\|F\|max 443 eV/Å) graphed vs eager e3nn is 1.2-1.9e-11. This is the same size as eager hybrid/flash vs eager e3nn (5.9e-12 to 1.5e-11), and graphed vs eager of the same backend is ≤9.0e-12 (relative ≤4e-14). In other words this is fp64 noise that scales with \|F\|. After two recaptures, allocated memory is 161 MiB (hybrid) / 128 MiB (flash) higher, consistent with the larger capacity. |
| 3a | Zero edges (single C, far O-H pair) | PASS | - | Energy bitwise equal to eager e3nn, forces exactly 0. |
| 3b | E exactly at capacity (E_cap=E=301), capacity+1 (E=302 → recapture to 384), then back to 301 | PASS | - | \|dF\| ≤5.0e-15, \|dE\|=0, exactly one recapture. |
| 3c | Periodic wbox648 (4 geometries, E 35660-35720) | PASS | - | hybrid \|dF\| ≤1.8e-14, flash ≤1.2e-14, dE 0 |
| 3d | 4 graphed models alive at once (aspirin and water192 × hybrid and flash), stepped in the order A,B,A,B,B,A | PASS | - | 24 comparisons, \|dF\| ≤7.5e-15. Replaying one graph 3 times leaves another graph's outputs bitwise unchanged. Constant-memory uploads inside each graph keep layers and models isolated from each other. |
| 3e | `step()` with a different system of the same N but different species | WRONG RESULT, silent | MEDIUM (API hazard) | `node_attrs`/`atomic_numbers` are captured statics and `step()` never refreshes or checks them. Aspirin with species reversed returned \|dE\|/atom 6.7 eV and \|dF\| 308 eV/Å with no error. The fix is to assert the static per-node tensors match, or copy them in. |
| 3f | `step()` with a different N | INCONSISTENT | LOW | If E ≤ E_cap the shape mismatch raises an error. If E > E_cap, `step()` silently recaptures for the new system: the result is correct, but `self.N` stays stale (21 against 192 rows), and stepping the original system afterwards raises. With a stale larger N, the padding indices `k % N` could go out of range for a smaller new system. |
| 3g | Returned tensors are views into static buffers | as documented | INFO | The next `step()` overwrites them, as the docstring says. Callers must `.clone()`. |
| 4 | Eager parity of the graphcap build vs cgpatch on real MACE layer inputs (aspirin, water192, wbox648, both layers): out, d_in1/2/w, dd_in1/2/w | PASS | - | 48 tensors, max relative difference 4.9e-15. 27/48 are bitwise equal (all d_in2, d_w, dd_in2, dd_w; out and *_in1 differ only by atomic ordering). Full-model eager flash: dE=0, \|dF\| ≤2.6e-15. The upstream `example/flashTP_test.py` fp64 checks (L0 E4096, L1 E4096, L1 E32768) all pass at strict rtol=atol=1e-10 (max abs 2.4e-12, on dd_in1). |
| 5a | Graphed timings | REPRODUCED (within 3%) | - | See the table below. Graphed hybrid median: aspirin 1.40 ms (claim 1.44), water192 2.96-2.97 (claim 3.01), water648 7.69-7.71 (claim 7.75). |
| 5b | Eager baselines | NOT REPRODUCED, about 10% lower here | LOW | Eager hybrid ran 15.2-15.9 ms here vs the claimed ~17. Eager flash on aspirin ran 9.6-9.8 vs the claimed 11.1. Eager is CPU-launch-bound (about 600 launches, 1.2 ms of GPU work), so it is sensitive to host state. The claimed speedups are therefore about 7-10% too high, e.g. aspirin hybrid model-only 10.9x here vs 11.7x claimed. Graphed hybrid still wins clearly at every size. |
| 5c | "graph pool MiB" column in SUMMARY_run2_tables | MISLABELED | LOW | The value is the peak allocation during capture, including warmups, and it matches my `peak_incl_capture` exactly (e.g. 1296 MiB, water648 hybrid). What stays resident afterwards is smaller: the allocated delta is 64-128 MiB and the reserved-pool delta is 74-208 MiB for hybrid/flash, 518-1728 MiB for e3nn. |
| 5d | Capture cost | mostly REPRODUCED | INFO | 0.05-0.17 s per capture/recapture. The first hybrid capture in a process costs 0.57-0.72 s because of cueq JIT/TorchScript warmup; SUMMARY lists 0.12 s for aspirin hybrid. |
| 5e | Padding overhead | INFO | INFO | The replay does more GPU work than eager because of the 20% padded edges: water648 hybrid 7.47 vs 6.62 ms summed kernel time. At large N this overhead is not hidden. A smaller headroom, or bucketing, would recover part of it. |
| 6 | Replay determinism | INFO | INFO | Two replays on identical inputs differ by up to 5.8e-12 eV/Å at \|F\|=443, and by about 1e-15 at normal geometries. This comes from the atomic scatter and is the same in eager. |

Pass thresholds used: graphed vs eager e3nn \|dE\|/atom < 1e-10 eV and max\|dF\| < 1e-11 eV/Å. For the compressed geometries the criterion is graphed vs eager same backend < 1e-11 and relative < 1e-13. My script's own built-in relative cutoff of 1e-14 was too strict, which is why the recapture log prints "all ok False". Every recaptured row is within fp64 noise.

## Timing (ms, median [p10-p90], 15 warmups, 80 reps, cuda-synced). Run A = job 1024, run B = job 1025

| system | backend | E / E_cap | eager model A | eager model B | graphed copy-in+replay A | B | replay only A | full step eager (GPU build) A | full step graphed (GPU build) A | B | capture s | peak during capture MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspirin | hybrid | 302/384 | 15.16 [15.12-15.30] | 15.35 [15.26-15.55] | 1.40 [1.40-1.40] | 1.40 [1.40-1.40] | 1.34 | 15.72 | 1.84 [1.82-1.86] | 1.85 | 0.57 (first in process) | 154 |
| aspirin | flash | 302/384 | 9.64 [9.61-9.69] | 9.79 [9.74-9.88] | 1.97 [1.96-1.97] | 1.97 | 1.90 | 10.14 | 2.41 [2.40-2.42] | 2.42 | 0.05 | 138 |
| water192 | hybrid | 6634/8000 | 15.74 [15.63-16.00] | 15.70 [15.56-15.92] | 2.97 [2.96-2.97] | 2.96 | 2.91 | 16.27 | 3.43 [3.42-3.45] | 3.46 | 0.07 | 390 |
| water192 | flash | 6634/8000 | 10.11 [10.05-10.19] | 10.10 [10.06-10.16] | 6.65 [6.64-6.68] | 6.64 | 6.58 | 10.67 | 7.11 [7.10-7.16] | 7.12 | 0.05 | 714 |
| water648 | hybrid | 26550/31872 | 15.64 [15.57-15.82] | 15.85 [15.77-16.14] | 7.69 [7.68-7.70] | 7.71 | 7.62 | 16.17 | 8.18 [8.16-8.21] | 8.21 | 0.08 | 1296 |
| water648 | flash | 26550/31872 | 20.71 [20.69-20.74] | 20.70 [20.69-20.75] | 19.64 [19.63-19.66] | 19.72 | 19.58 | 21.23 | 20.14 [20.12-20.17] | 20.25 | 0.09 | 2278 |

e3nn reference times are in the JSON (graphed e3nn: aspirin 2.55 ms, water648 38.25 ms vs 36.36 ms eager). GPU graph build alone: 0.40-0.47 ms. The two runs agree to within 1.5% everywhere.

## Verdict
The correctness claims hold: fp64 parity with eager e3nn, exact zero-edge handling, padding that is exactly inert, correct recapture, and graphs isolated from each other when several are alive. The FlashTP graphcap build matches cgpatch numerically. The graphed latencies reproduce. The eager baselines were about 10% slower in the prototype's run than here, so its quoted speedups are slightly inflated. The one real defect is an API hazard: `step()` neither validates nor refreshes the per-node statics (species, N). That is MEDIUM severity for any MD driver that might reuse a GraphedMACE across systems.
