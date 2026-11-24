# FlashTP adapter v2 / hybrid v2: limitations and design choices

Experimental. Validated only for MACE 0.3.17 + e3nn 0.4.4 + torch 2.11.0+cu128 + FlashTP cgpatch (`src/flashTP-cgpatch`, 0e4d820) on H100, MACE-OFF23 medium, fp64.

## state_dict / checkpoints (item 5): chosen behaviour
- `FlashConv` keeps the original e3nn `conv_tp` as the submodule `orig`. It never calls it.
- A state_dict hook removes the `orig.` prefix and drops anything under `ftp`, which carries no state.
- While installed, `model.state_dict()` therefore has exactly the stock MACE keys, in stock order, with stock `_metadata` keys and bitwise-identical values.
- A load_state_dict pre-hook adds the prefix back, so `model.load_state_dict(stock_sd, strict=True)` works while installed.
- A state_dict saved while installed loads strictly into an unmodified MACE model.
- Pickling a `FlashConv` raises `AdapterError` ("uninstall first"). This covers `torch.save(model)` and `copy.deepcopy(model)` of the whole module. The reasons:
  - the pickle would reference this adapter's classes;
  - it would carry device-pinned FlashTP metadata outside the state_dict;
  - a reload would not work without FlashTP.
- To save a whole model, call `uninstall(model)` first, or save `model.state_dict()`.
- `sc="cueq"` (`CueqSC`) does **not** preserve keys: `products.*.symmetric_contractions.*` take the cuet layout while installed. `hybrid_v2.uninstall_cueq_sc(model)` restores the original modules.
- Install bookkeeping is a plain Python attribute on the model (`_flashtp_v2_saved` / `_cueq_sc_v2_saved`). It is never in the state_dict, and uninstall removes it.

## Install / uninstall
- `install(model)` validates every selected block first and changes nothing if any check fails. A second install raises `AdapterError`. `uninstall(model)` is idempotent: it returns True once, then False. It restores the identical module objects and removes `conv_fusion`.
- With `hybrid_v2`, install the cueq SC first. `build_calc` does this; the conversion deep-copies the model, which a FlashTP-installed model refuses.
- `model.to(...)`, `.float()` and `.double()` while installed convert `orig` but not the FlashTP kernel metadata. The next forward then raises a dtype/device mismatch. Uninstall, convert, and reinstall instead.

## Supported models
- Exact classes (MACE 0.3.17 `mace.modules.blocks`) whose `forward` has the `conv_fusion` branch:
  - RealAgnosticInteractionBlock
  - RealAgnosticResidualInteractionBlock
  - RealAgnosticDensityInteractionBlock
  - RealAgnosticDensityResidualInteractionBlock
  - RealAgnosticAttResidualInteractionBlock
  - RealAgnosticResidualNonLinearInteractionBlock
- Only the first two are exercised by MACE-OFF23 and tested.
- `conv_tp` must be a plain e3nn `o3.TensorProduct` with these properties:
  - uvu and weighted instructions;
  - external per-edge weights;
  - e3nn-default path weights;
  - in1 multiplicities that are multiples of 32.
- The following are rejected: subclasses, Magnetic* blocks, cueq/oeq-enabled models, blocks with `conv_fusion` already set, and MACE versions other than 0.3.17.

## Runtime
- **Single stream / single thread.** Kernels run on the current stream of the build device, and the adapter has no locking. Do not share one installed model across threads or CUDA streams. The device must match the build device: `FlashConv` checks this, and `install` follows the model's parameter device.
- **No TorchScript, no LAMMPS/ML-IAP, no torch.compile.** The module is Python-only. `use_lammps` (FlashTP's `_lammps` kernels) is not used, and the MACE calculator must use `use_compile=False`, the default here. MACE's padded-graph compile path was not tested.
- **Not bitwise deterministic.** The FlashTP scatter uses `atomicAdd`, so repeated evaluations differ at fp64 rounding level (about 1e-13 eV/atom, about 1e-15 eV/Å). e3nn's `scatter_sum` (`index_add` on CUDA) is not bitwise deterministic either.
- **Index checks.** With the default `check_indices=True`, each FlashConv call costs one `aminmax` over `edge_index` plus one host sync per interaction layer. Pass `check_indices=False` (`build_calc(..., check_indices=False)`) to skip it. Without the check, out-of-range indices are undefined behaviour in the kernel.
- The int64→int32 conversion of `edge_index` happens on every call; there is no caching (v0's cache served stale graphs). `n_nodes` and `n_edges` must be below 2^31, and this is checked.
- **Zero edges** (E=0 or N=0) never launch the kernel; the kernel fails with "invalid configuration argument" at E=0. The result is zeros `[N, irreps_out.dim]`, and the gradients are zeros of the input shapes, as with e3nn. Graphs with some isolated nodes (E>0) go through the kernel normally.
- **extcg fp32 truncation.** In the FlashTP extcg fp64 kernel (`fwd_sptp_linear_v2_shared_exp_double.cu:137`), `reg_w_path_norm` is declared `float`, which limits that path to about 2.5e-8 relative accuracy. The extcg kernel is chosen only when a layer has more than 256 unique CG values. MACE-OFF23 medium uses the `large` kernel in both layers (`kernel_class` in the info/results), so this does not apply here. It would apply to higher-l models (e.g. nequip 4_32_5).
- CG fp64 dedup tolerance (1e-12) was validated for l ≤ 6 only. MACE-OFF23 medium has l ≤ 3.
- fp32 (`dtype=torch.float32`) builds but was not parity-tested in v2.
