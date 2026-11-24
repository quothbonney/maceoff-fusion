# FlashTP fp64 CG patch: notes

- Branch `upstream/fp64-cg-minimal` in worktree `src/flashTP-cgpatch`, commit `0e4d820cc292161051b8e95f1e9a32bcb8c8bc6b`, based on upstream `d52e414`.
- Patch file: `results/review/0001-Keep-exact-fp64-CG-values-in-fp64-mode-CG-precision-.patch`
- Diffstat: 4 files, +40/-9. Changes are in `flashTP.py` (+43/-6) and one line each in `fused_e3nn_exp_opt_large.py`, `fused_e3nn_exp_opt_large_lammps.py` and `fused_e3nn_exp_opt_extcg.py`.
- Author is set to `maceoff-study review <review@localhost>`. The repo-local git config carries the upstream maintainer's identity, so it was not used. Amend the author before sending upstream.
- Nothing was pushed or published.

## Problem
When `dtype=torch.float64`, FlashTP output and its 1st and 2nd derivatives match e3nn fp64 only to 2e-8 to 2e-7 relative, which is about fp32 CG accuracy. The fp64 kernels already store the CG table as `__constant__ double unique_cg_val[]`.

## Root cause
1. `extract_cg_info()` rounds every `wigner_3j` value to fp32 before it builds the unique-value table. The intent was to merge fp64 values that are equal except for rounding noise. `cgmat2fiber()` then looks values up through the same fp32 rounding, so the double table only ever holds fp32-rounded numbers. The max |CG − exact| is 1.0e-8 to 1.5e-8.
2. `wigner_3j()` is called without `dtype`, so the CG precision follows `torch.get_default_dtype()` instead of the `dtype` argument. As a result, the experimental fix on `exp/fp64-cg` (02de84c) still gives fp32 CG (1.0e-8 to 1.5e-8 error) when a caller builds the module with default dtype float32 and `dtype=float64`. Upstream's example and our MACE adapter both set the default dtype first, which hides this.

## Fix (fp64 only)
- When `dtype == float64`, `extract_cg_info()` calls `wigner_3j(..., dtype=torch.float64)`. It merges values that agree within `_CG_FP64_DEDUP_TOL = 1e-12` (sorted single pass) and snaps every CG matrix entry to its table representative, keeping exact zeros. The CG precision now follows `dtype`.
- `cgmat2fiber()` in large, large_lammps and extcg looks values up with `.to(self.dtype)` instead of `.to(torch.float32)`. For fp32 this is the same operation, and the exact lookup still works because the entries were snapped.
- The fp32 branch of `extract_cg_info()` is the original code, unchanged.

## Why the tolerance is safe (`scripts/review/cg_table_analysis.py` → `cg_table_analysis.json`)
Measured on e3nn 0.4.4's fp64 `wigner_3j`, which comes from stored fp64 constants:

| set | distinct values | max spread of equal values | min gap between distinct values |
|---|---|---|---|
| all triples l ≤ 3 | 44 | 8.9e-16 | 3.5e-3 |
| all triples l ≤ 5 | 339 | 8.9e-16 | 1.3e-4 |
| all triples l ≤ 6 | 732 | 8.9e-16 | 5.5e-6 |

- The cluster count is the same for every tolerance from 1e-15 to 1e-6.
- No mathematically-zero values appear as tiny nonzeros.
- For every layer in `ir_config/{mace,nequip,sevennet}`, the unique count after dedupe equals upstream's fp32-rounded count. The large vs extcg choice (256-entry uint8 limit) is therefore unchanged. For example, nequip `4_32_5` L1 has 340 values and uses extcg in both.
- fp32 rounding never merges distinct values up to l = 6.

## Evidence
- GPU: H100 80GB HBM3, uuid GPU-4b202516-7dca-26da-e0bf-7f5aef5ca845.
- SLURM jobs: 995 (main), 998 (empty-edge and repeatability checks), 999 (fp32 metadata).
- Setup: 64 nodes × 64 edges, random inputs, unsorted dst, reference = e3nn fp64 TensorProduct plus scatter.
- Metric: worst over {out, ∂/∂(in1, in2, w), 2nd-order grads}, as max abs err / max |ref|.
- Source: `results/review/kernel/kernel_parity_*.json`.

| layer (path) | upstream fp64 | exp/fp64-cg, default fp64 | exp/fp64-cg, default fp32 | **this patch** (either default) |
|---|---|---|---|---|
| mace/medium L0 (large) | 1.9e-8 | 1.3e-15 | 1.9e-8 | **1.4e-15** |
| mace/medium L1 (large) | 1.6e-7 | 5.4e-15 | 1.6e-7 | **5.4e-15** |
| sevennet-l3i5 L1 (large) | 1.6e-7 | 2.3e-14 | 1.6e-7 | **2.3e-14** |
| nequip 4_32_5 L1 (extcg) | 2.0e-7 | 2.5e-8 | 2.0e-7 | **2.5e-8** (see limitations) |

- Forward error of the output alone with this patch: 1.3e-15 to 2.3e-15 on the large path.
- CG table check, run on CPU for every tree and default-dtype combination (`cg_table_tree_summary.txt`): with this patch, max |CG − exact| is 6.1e-16 to 8.9e-16 for fp64 under either default dtype.
- Upstream `example/flashTP_test.py` (32768 edges) on the patched tree prints 7/7 "All Good" in both fp64 and fp32, for mace/medium L0 and L1 and sevennet-l3i5 L1. Logs: `logs/review/example_cgpatch_*`.

## fp32 unchanged
- On GPU, the full fp32 module metadata (fiber arrays, CG table, path weights, block config) is bitwise identical between upstream and the patch. This holds for all four layers under default dtype fp32 and fp64 (`fp32_metadata_bitwise.txt`).
- The CPU table and fiber-index fingerprints are also identical (`cg_table_tree_summary.txt`).
- fp32 outputs are not bitwise reproducible even run to run within one tree, because the kernels use `atomicAdd`. The differences between upstream and patch fp32 outputs (≤ 5.5e-7 relative) are at that noise level, and the fp32 errors against the reference match (about 4e-7).

## Limitations / follow-ups
- **extcg fp64 path remains about 2.5e-8 relative.** This is not caused by the CG table. `sptp_exp_opt_extcg/fwd_sptp_linear_v2_shared_exp_double.cu:137` declares `float reg_w_path_norm = weight[..] * path_weight[..];`, which truncates weight × path norm to fp32 in the fp64 forward pass. The fix is to change it to `double` (or `scalar_t`). It needs a kernel rebuild, so it is not included or tested here. It is the only `float` temporary in any `*_double.cu`.
- The legacy `sptp_exp_opt/fused_e3nn_exp_opt.py` has its own copy of the fp32 rounding. `uvu_TP` does not use it, so it was left untouched.
- The tolerance was validated for l ≤ 6. e3nn ships constants up to l = 11. At higher l, re-check the minimum gap; a larger l only shrinks the gap.
- Construction cost is unchanged: the same O(#nonzeros × #unique) `list.index` lookup.
