# Independent review 2: flash_v2 adapter/hybrid + FlashTP fp64-CG patch (0e4d820)

Verdict: **no blockers and no majors.** Every claim I tested held. The findings below are minor items and notes.

## Setup
- Jobs: SLURM 1009 and 1010 on H100 GPU-884d19b4-3b1e-5246-0d07-a1a68b778800, and 1011 on GPU-4b202516-7dca-26da-e0bf-7f5aef5ca845. All jobs finished; none are left running.
- Scripts: `scripts/review2/review2.py` (one process per global default dtype), `compare_defaults.py`, `run_review2{,b,c}.sh`.
- Logs: `logs/review2/run{1,2,3}.log`.
- Evidence:
  - `results/review2/review2_default_float{64,32}.json/.npz` (run 1, all tests)
  - `review2b_*` (graph_changes, zero_edge and cleanup rerun after I fixed the harness criteria)
  - `review2c_*` (zero-edge training gradients with parameters set to `requires_grad`)
  - `compare_defaults.json`
  - `summary.json` (condensed)
- Run 1 had three harness bugs, all on my side:
  - It used absolute thresholds on clashing geometries.
  - It required bitwise e3nn repeatability, which e3nn's `index_add` does not give.
  - Parameters had `requires_grad=False`.
  - The b and c runs fix these. None of them was an adapter defect.
- I modified nothing under review.
- Model and settings for every run: MACE-OFF23 medium, fp64 unless stated otherwise. Every check ran under global default float64 **and** float32.

## Findings

| # | Area | Result | Severity | Evidence |
|---|---|---|---|---|
| 1 | fp64 parity, non-water molecules (CH3CH2OH, CH3COOH, C6H6, CH3SH, CH3Cl), flash and hybrid vs e3nn | pass. dE/atom 0; max\|dF\| ≤ 3.1e-15 eV/Å | – | `parity_fp64` |
| 2 | fp64 parity, **periodic** water box (648 atoms, 35198 edges) and strained triclinic box (192 atoms), including **stress** | pass. Worst over both defaults: dE/atom ≤ 1.5e-13 (hybrid); max\|dF\| ≤ 6.2e-15; RMS ≤ 1.1e-15; max\|dS\| ≤ 2.3e-17 eV/Å³ (\|S\| ≈ 5e-3 to 1.6e-2) | – | `parity_fp64`, `summary.json` |
| 3 | `build_calc` hard-codes `compute_stress=False`, so `atoms.get_stress()` raises `PropertyNotImplementedError`. Stress is correct through `model(..., compute_stress=True)` (row 2). Neither READY nor LIMITATIONS mentions this, and the authors' tests never exercise stress. | functional gap | minor | `parity_fp64.ase_get_stress_flash` |
| 4 | fp32 build (`dtype=float32`), flash and hybrid | runs; errors are at fp32 noise. water648: flash32 vs e3nn32 max\|dF\| 1.4e-6, against e3nn32 run-to-run 1.3e-6. dE/atom 0 to 4.8e-5, which is one fp32 ulp of E≈-1.3e5 to -4.5e5 eV divided by N; e3nn32 repeat shows the same 4.8e-5. Stress 1e-8. fp32 vs fp64 itself is 5e-4 eV/atom and 1e-4 eV/Å for every backend: a MACE fp32 property, not caused by the adapter. | – (note: fp32 now parity-checked) | `fp32` |
| 5 | Default-dtype independence | pass. The fp64 CG tables and all FlashTP metadata tensors are **bitwise identical** between processes under default float32 and float64. The outputs of all systems differ by at most 4.5e-15, which is atomic noise. `cg_report` gives max\|CG−e3nn fp64\| 6.1e-16 (L0) and 7.2e-16 (L1); 3/5 and 17/19 of the values are not fp32-representable. | – | `compare_defaults.json`, `cg_tables` |
| 6 | Default dtype after `build_calc` exceptions (bad conv, bad path, cuda:7, float16, injected `FA.install` failure) | pass. The default is unchanged in every case. Note that this MACE 0.3.17 `MACECalculator` scopes its own dtype changes, so `build_calc`'s restore branch never runs. The final "default changed" check raises without restoring first, but it is unreachable today. | note | `build_errors` |
| 7 | Graph changes through the ASE path on one calc. Alternating systems: 192/648-atom periodic boxes, molecules, atom-permuted box (same edge count, different `edge_index`), strained cell. In-place edits on one Atoms object: positions (global and single-atom), numbers, cell. Edge counts vary 10380–10396. | pass. There are no stale results. Normal geometries give dE ≤ 3e-11 eV (\|E\|=1.3e5) and dF ≤ 7e-15. My single-atom 0.4 Å kicks created clashes (\|F\|≈5e7 eV/Å); there the error is at relative 1e-14, the same as e3nn's self-repeat on that geometry. | – | `review2b_*.graph_changes` |
| 8 | Multi-graph batch (ethanol + isolated Cl + 192 box + far pair + CH3SH) | pass. Per-graph energies match single-graph e3nn exactly (dE 0); max dF 8.9e-15 | – | `graph_changes.batch_*` |
| 9 | Zero-edge paths: single atom, 3 far atoms, periodic single O with stress, all-isolated batch | pass. dE 0, F and S exactly 0 | – | `review2c_*.zero_edge` |
| 10 | Zero-edge plus `training=True` (create_graph) with `loss = E + Σ F²` backpropagated to parameters (double backward through `_ZeroConv`), single Cl and mixed ethanol+Cl | pass. All 29 parameter grads exist in both backends; max diff 0 (single atom) and 1.4e-14 (mixed). No autograd errors. | – | `review2c_*.zero_edge.train_*` |
| 11 | Input checks: index = N, index < 0, fp32 inputs to an fp64 build, CPU inputs | pass: all four raise `AdapterError`. int32 `edge_index` and non-contiguous `node_feats` are accepted. | – | `input_checks` |
| 12 | 20 install/uninstall cycles alternating `check_indices` | pass. `memory_allocated` is constant at 151473664 B. 0/20 FlashConv objects survive gc. The second uninstall returns False each time. The original `conv_tp` objects are restored; no `conv_fusion` or state attribute is left; 0 state_dict hooks before and after. The state_dict and `_metadata` are bitwise equal to the stock ones, both while installed and after. `load_state_dict(strict=True)` works while installed. `torch.save` and `deepcopy` raise `AdapterError` while installed, and `torch.save` works after uninstall. | – | `review2b_*.cleanup` |
| 13 | Partial install failure (injected exception while building the layer-1 FlashConv, after layer 0 was built) | pass. The model is unchanged and `is_installed` is False | – | `cleanup.partial_*` |
| 14 | Hybrid → `FA.uninstall` + `uninstall_cueq_sc` | pass. The state_dict and metadata are bitwise equal to the stock model | – | `cleanup.hybrid_*` |
| 15 | Repeated `build_calc` (5× hybrid) in one process | pass. Memory flat; dE 0, dF ≤ 2.9e-15 | – | `repeat_build` |
| 16 | Run-to-run variation over 20 calls, water648 periodic box with stress | E bitwise stable (1 distinct value) for e3nn, flash and hybrid. Max dF is 3.8e-15 for flash vs 4.2e-15 for e3nn; dS ≤ 5e-17. | – | `run_to_run` |
| 17 | `check_indices` overhead (median of 40, interleaved, forward+forces) | water648 box: 22.42 vs 22.10 ms, +0.32 ms (+1.5%). Ethanol: +0.11 ms (+1.1%). Default `True` is sensible; turn it off only for timing runs, as documented. | note | `index_overhead` |
| 18 | While installed, `model.named_parameters()` / `parameters()` names contain `.orig.` (the hooks only fix up `state_dict`). Tools that key on parameter names see non-stock names: optimizer groups by name, EMA, fine-tuning freeze lists. My training test had to strip `.orig.`. | undocumented | minor | `t_zero_edge` code |
| 19 | Untested by me: multi-thread/multi-stream use, `model.to()` while installed (documented as unsupported), and the extcg path (not used by MACE-OFF). | not covered | note | – |

## Patch review (src/flashTP-cgpatch 0e4d820, `results/review/0001-*.patch`)

| # | Item | Result | Severity |
|---|---|---|---|
| P1 | fp32 path unchanged | pass (code read). The `else` branch of `extract_cg_info` is the upstream code verbatim. `cgmat2fiber` now uses `.to(self.dtype)`, which is the same operation as `.to(torch.float32)` when dtype=fp32. This matches the authors' bitwise fp32 metadata evidence. | – |
| P2 | fp64 correctness | pass. Every fp64 table value is within 7.2e-16 of e3nn fp64. MACE parity results are in rows 1–2. Tables are bitwise identical across default dtypes (row 5). | – |
| P3 | Dedup tolerance 1e-12 | sound. `_dedup_cg_fp64` makes a sorted single pass against the first member of each cluster, so it cannot chain-merge. Each value is within tol of its representative by construction, so the `assert` holds (it is also stripped under `python -O`, which is harmless here). Exact zeros are preserved. Spread of equal values ≤ 8.9e-16 vs minimum gap 5.5e-6 (l ≤ 6). For l > 6 the caveat is documented. | note |
| P4 | Edge dtypes | float16/bfloat16 take the old branch, and the lookup is now `.to(fp16)` instead of `.to(fp32)`. Upstream could not match an fp32 value against an fp16 table either, so this is no regression. The FlashTP kernels are float/double only. The adapter rejects anything except fp32/fp64. fp64 with a CPU device: the table code does not depend on device, and the kernels are CUDA-only regardless. | note |
| P5 | Pre-existing, not touched by the patch: the fp32 branch still calls `wigner_3j` without a dtype, so fp32 tables are fp64→fp32 or fp32-native depending on the default dtype. Possible last-ulp differences. | upstream behaviour | note |
| P6 | PATCH_NOTES accuracy | Mostly accurate, and the jobs and GPU uuid are consistent. One error: the notes say "flashTP.py (+43/-6)", but `git diff --numstat` gives **+37/-6** (43 is the changed-line total); the overall +40/-9 is right. The extcg `float reg_w_path_norm` limitation is correctly flagged. | minor (doc) |
| P7 | Author placeholder `maceoff-study review <review@localhost>`; nothing pushed | as intended (left unchanged per user) | note |

## Recommended follow-ups (all optional)
1. Add a `compute_stress` argument to `build_calc`, or document the workaround: `calc.compute_stress=True` alone is not enough, because `implemented_properties` has already been filtered.
2. Document the `.orig.` parameter-name difference in LIMITATIONS.md.
3. Fix the flashTP.py diffstat in PATCH_NOTES.md.
4. Update LIMITATIONS.md to say fp32 is now checked. It is at fp32 noise (row 4).
