# Phase 7 interfaces (coordinator-owned; agents must not edit this file)

Root: R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study. `source $R/env.sh`. Python: $R/venv-cueq/bin/python -I.
Env for every GPU job (same as scripts/cudagraph6/*):
  SP=$R/venv-cueq/lib/python3.11/site-packages
  export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
FlashTP: graph-capture build src/flashTP-graphcap (exp/graph-capture @ 415e7a3, fp64 CG, in-place .so). Backends are built
through scripts/cudagraph6/hybrid_gc.py (import it FIRST; it pins flashTP_e3nn to src/flashTP-graphcap and asserts fp64 CG).
GPU: `srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash <script>` -- one H100 per job; log SLURM_JOB_ID + nvidia-smi uuid; never leave jobs running.
No pip installs. Do not modify src/flashTP, src/flashTP-fp64cg, src/flashTP-cgpatch, src/flashTP-graphcap, scripts/{flash_proto,hybrid,flash_v2,graph,cudagraph6}/*, venvs, or any existing result/log.
Systems: results/workloads/systems/*.extxyz (aspirin, org192, water192, org648, water648, wbox192, wbox648 [periodic]) + ase.build.molecule("H2O")
+ zero-edge cases (single C atom; two O atoms 20 A apart). Model: $R/models/MACE-OFF23_medium.model (fp64). MACE 0.3.17.

## Ownership (each agent writes only in its own dirs: scripts/<dir>, results/<dir>, logs/<dir>)
- A  graphed7     : corrected graphed runner
- B  fuse7_embed  : fused edge-embedding kernel(s) (spherical harmonics and/or radial basis+cutoff, optionally edge vectors/lengths)
- C  fuse7_node   : fused post-convolution node kernel (interaction linear + 1/avg_num_neighbors + skip_tp fold, per species)
- D  validate7    : profiling of corrected graphed path, then integrated correctness + performance gate

## Fusion module contract (B and C)
`scripts/<dir>/<module>.py` must expose:
  install(model) -> handle      # in-memory module swap on a stock-loaded MACE model (after hybrid/flash adapters are installed);
                                # must NOT edit installed MACE sources; must raise a clear error on unsupported models.
  uninstall(model, handle)      # restores exact original modules; idempotent
  describe() -> dict            # which MACE ops are replaced, kernel names, launches per fwd/bwd
- Implementation: real fused GPU kernels (Triton 3.6 from venv-cueq, or CUDA via torch.utils.cpp_extension.load_inline with the
  contained CUDA 12.8 nvcc; build dir under $TORCH_EXTENSIONS_DIR / $TRITON_CACHE_DIR which env.sh points inside R).
- fp64 forward AND a custom backward (torch.autograd.Function) giving first derivatives w.r.t. every differentiable input that
  MACE's force (and stress: positions/shifts via displacement) path needs. Inference only: create_graph=False; double backward may raise
  a clear NotImplementedError.
- Must be CUDA-graph capturable: no host syncs, no allocations that depend on data values, launches on torch's current stream,
  shapes depend only on (n_nodes, n_edges). Must handle n_edges = 0.
- Write results/<dir>/READY (import recipe + what is fused + local test results) once local parity passes.

## Graphed runner contract (A)
`scripts/graphed7/graphed_runner.py`:
  build_backend(kind in {"e3nn","flash","hybrid"}, transforms=()) -> (calc, model)   # transforms: callables model->handle applied before capture
  GraphedRunner(calc, model, *, edge_margin=0.20, mult=64, stress=False)
  runner.step(atoms) -> dict(energy=float, forces=np.ndarray[N,3], stress=np.ndarray[6] or None, recaptured=bool, E=int, E_cap=int)
  runner.stats() -> dict(captures=int, recaptures=int, last_capture_s=float, key=...)
- A graph is keyed to (n_atoms, atomic numbers in order, pbc, cell-is-None, head, dtype, stress flag). ANY mismatch -> either
  raise a clear error or transparently build a separate graph for the new key (document which); never replay a graph for a different key.
- Graph build via scripts/graph/proto_graph.py (GPU) by default; option to use stock MACE batch for validation.
- Write results/graphed7/READY when its tests pass.

## Correctness gate (all new paths vs eager stock e3nn MACE, fp64)
Local op outputs + grads; whole-model |dE|/atom, max |dF|, RMS dF (eV/A), stress |dsigma| (eV/A^3) for periodic if supported;
H2O, aspirin, ~200, ~650, zero-edge, periodic box; changing coordinates + neighbour lists; repeated calls; recapture.
Thresholds for fp64: |dE|/atom <= 1e-11, max|dF| <= 1e-11 (report actual values). Any silent wrong result = blocker.
