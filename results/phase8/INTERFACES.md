# Phase 8 interfaces: node-centric FP64 TP convolution (coordinator-owned; do not edit)

Root R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study. `source $R/env.sh`. Python $R/venv-cueq/bin/python -I.
GPU job env (copy scripts/validate7/job.sh): LD_LIBRARY_PATH to venv-cueq nvidia libs, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1,
PATH=$R/venv-cueq/bin:$PATH (ninja for load_inline). CUDA 12.8 nvcc in $CUDA_HOME (env.sh). Nsight Compute CLI:
$R/toolchain/cuda-12.8.1-extract/nsight_compute/ncu (may need GPU perf-counter permission; if ERR_NVGPUCTRPERM, report it).
GPU: `srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash <script>`; one H100 per job; record SLURM_JOB_ID + nvidia-smi uuid; end all jobs.
No pip installs into venv or venv-cueq. Isolated new venvs allowed only by agent A (venv-oeq). Do not modify src/flashTP*,
scripts/{flash_proto,hybrid,flash_v2,graph,cudagraph6,graphed7,fuse7_embed,fuse7_node,validate7}/*, venvs, or prior results.
Each agent writes only to scripts/<dir>, results/<dir>, logs/<dir> (and its own git branch/worktree if any).

Agents / dirs: A oeq8 (existing backends), B nodetp8_fwd (forward kernels), C nodetp8_bwd (backward kernels, autograd op,
correctness), D prof8 (Nsight Compute FlashTP profile, then integration + benchmarks).

## Problem definition (MACE-OFF23 medium, fp64; see results/mace_layer_inventory.json)
Per interaction layer i, given node features x [N, d1] (after linear_up), edge attrs y [E,16] (real SH l<=3, e3nn component
normalisation), per-edge weights w [E, W] (radial MLP output), edge_index [2,E] int64 (row0 = sender/src, row1 = receiver/dst):
    out[n] = sum_{e: dst(e)=n} TP_uvu(x[src(e)], y[e], w[e])   -> [N, d_out]   (e3nn mul_ir layout, MACE conv_tp semantics)
Reference = MACE's own model.interactions[i].conv_tp (e3nn o3.TensorProduct, fp64) + gather + scatter_sum (blocks.py).
L0: x 128x0e (128), y 1x0e+1x1o+1x2e+1x3o (16), out 128x0e+128x1o+128x2e+128x3o (2048), 4 uvu instr, W=512.
L1: x 128x0e+128x1o (512), out 10 irreps of 128 (5120, sorted irreps_out, not simplified), 10 uvu instr, W=1280.
Weights layout: [E, W], instruction-major chunks of 128 (u index) in conv_tp.instructions order. Path weights / CG: take from the
e3nn module exactly (conv_tp.instructions[k].path_weight, o3.wigner_3j(l1,l2,l3, dtype=float64)) -- FULL fp64 CG, never fp32-rounded.
Edges may include padding self-edges with zero weights (graphed runner); E may be 0; N >= 1. Edges arrive sorted by src, NOT by dst.

## Graph format for node-centric kernels (shared; implemented by C in scripts/nodetp8_bwd/csr.py, B may stub until then)
build_csr(edge_index, N) -> dict(perm_dst [E] int32/int64 (edges sorted by dst, stable), rowptr_dst [N+1] int32,
                                   perm_src [E], rowptr_src [N+1])  -- GPU only, no host sync, CUDA-graph capturable (torch.sort/
                                   searchsorted/bincount-with-minlength+cumsum on fixed shapes).
Built once per forward call and shared by both layers and the backward.

## Kernel API
B: scripts/nodetp8_fwd/nodetp_fwd.py:  fwd(layer_cfg, x, y, w, edge_index, csr) -> out [N, d_out]
C: scripts/nodetp8_bwd/nodetp_bwd.py:  bwd(layer_cfg, grad_out, x, y, w, edge_index, csr) -> (gx [N,d1], gy [E,16], gw [E,W])
   layer_cfg = make_cfg(conv_tp) (C owns, B may provide an equivalent; agree on fields: irreps, instructions with path_weight,
   per-path CG sparse lists (fp64), offsets in x/out/w).
C also owns: NodeConvFunction (torch.autograd.Function: forward=B.fwd (or reference until ready), backward=C.bwd),
   install(model)/uninstall(model, handle) module swap mirroring scripts/flash_v2/flashtp_adapter_v2.py semantics
   (sets block.conv_fusion, replaces conv_tp; conv_tp(node_feats, edge_attrs, tp_weights, edge_index) -> message [N, d_out]);
   must coexist with graphed7 runner transforms and fused_embed / fused_node installs.
Kernels: fp64, launches on torch current stream, no host syncs, shapes depend only on (N, E), CUDA-graph capturable, E=0 ok.
Implementation choice free: CUDA via torch.utils.cpp_extension.load_inline (build dir under $TORCH_EXTENSIONS_DIR) or Triton.

## Correctness thresholds (fp64)
Local op vs reference: max abs and max abs/max|ref| <= 1e-12 for out, gx, gy, gw; report also near-zero outputs (|ref|<1e-8).
Whole model vs eager stock e3nn: |dE|/atom <= 1e-11 eV, max|dF| <= 1e-11 eV/A, stress |dsigma| <= 1e-14 eV/A^3.

## Fair comparison baselines (identical real inputs captured from model on: aspirin, org192, water192, org648, water648, wbox648)
FlashTP-FP64CG graphcap conv (src/flashTP-graphcap via scripts/cudagraph6/hybrid_gc.py), e3nn conv, cueq fused conv (MACE
enable_cueq, if separable), OEQ (agent A), new node-centric. Conv-only fwd, bwd, fwd+bwd; eager and under CUDA-graph replay.
Whole model: latest graphed hybrid + fused_embed + fused_node (scripts/graphed7 + fuse7_*; results/validate7) vs same with the
new conv replacing FlashTP.
