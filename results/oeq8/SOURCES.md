# oeq8 sources (agent A) — 2026-11-10

## LunaTP / LunaEquivariance
- Paper: https://arxiv.org/abs/2610.18393 (v1, 13 Oct 2026), HTML full text https://arxiv.org/html/2610.18393 .
  Node-centric (dst-CSR) execution, path-group parallelism, sparse expansion + product/coefficient reuse; bwd reduces over src-CSR;
  fwd, bwd, double- and triple-backward; FP32 and FP64 (App. D.1: FP64 means 4.4x fwd / 4.4x bwd over OpenEq); supports uuu/uvu/uvw.
  Paper's training numbers are on MACE-L/NequIP/SevenNet etc. in mostly FP32 -- not transferable to our MACE-OFF23-medium fp64 inference.
- Code: https://github.com/HPC-AI-Team/LunaEquivariance  (public, BSD-3-Clause, created 2026-10-11, pushed 2026-11-07)
  Pinned clone: $R/src/lunatp @ 0ead67d1536067960d87cfa4f83644501be8a320 ("Unify contraction input layout and tune CUDA kernels (#2)"), tag v0.2.0 exists.
  PyPI: LunaEquivariance 0.2.0; deps torch>=2.8,<3, numpy, ninja, e3nn>=0.5.2,<0.7 (conflicts with mace-torch 0.3.17's e3nn==0.4.4 pin ->
  installed --no-deps into $R/venv-luna; conv works with e3nn 0.4.4).
  API: TensorProductConv(e3nn_tp | irreps..., dtype=float64); make_graph(src, dst_sorted, num_nodes) -> LunaTPGraph; conv(x, edge_attr, weight, graph=g).
  Layout mul_ir; weights e3nn [E, W]; CG fp64 (own port of e3nn 0.6 wigner_3j, literals "%.17e"); graph = CSR by dst (edges must be dst-sorted)
  plus src-CSR/segment tables built on GPU (cub), but with a host readback (status.to(CPU)) -> not capturable. JIT via torch cpp_extension.
  Also ships o3.SymmetricContraction, Linear, SH, etc. (not evaluated here).

## OpenEquivariance
- https://github.com/PASSIONLab/OpenEquivariance ; PyPI openequivariance 0.8.0 (sdist inspected in $R/tmp/oeq8dl/oeqsrc). Paper: Bharadwaj et al. 2025.
- MACE 0.3.17 integration (venv-cueq/.../mace/modules/wrapper_ops.py, cli/convert_e3nn_oeq.py, calculators/mace.py enable_oeq):
  oeq.TPProblem(irreps_in1, irreps_in2, irreps_out, instructions, shared_weights=False, internal_weights=False, irrep_dtype=weight_dtype=
  oeq.torch_to_oeq_dtype(default)); oeq.TensorProductConv(tpp, deterministic=False) ONLY ("atomic"); forward(X, Y, W, rows=receiver, cols=sender).
  Weights passed in e3nn layout unchanged (verified numerically for uvu). Works with oeq 0.8.0 (no torch pin; JIT with nvcc from CUDA_HOME).
- Layout mul_ir (ir_mul accepted for pure uvu in 0.8.0). CG: e3nn_lite.wigner_3j fp64 * normalization, emitted as hex-float literals (exact fp64).
- Graph: COO int64 rows/cols. atomic mode: any order. deterministic mode: rows (=dst) row-major sorted + sender_perm (transpose permutation);
  fixup kernel merges warp partials; no shared-weights; kahan only fp32.
- Derivatives: fwd, bwd, double-bwd (both modes).

## FlashTP (local clones)
- $R/src/flashTP @ d52e414 (https://github.com/SNU-ARC/flashTP); $R/src/flashTP-graphcap @ 415e7a3 (our graph-capture + fp64-CG patches).
  Edge-centric, scatter by atomicAdd (no sorting required), COO int32 indices, mul_ir/e3nn layout; stock CG table fp32-rounded (patched to fp64
  in cgpatch/fp64cg/graphcap). fwd + bwd (+ double bwd in upstream).

## cuEquivariance (venv-cueq, 0.10.0)
- MACE conv_fusion: cuet.SegmentedPolynomial(channelwise_tensor_product(...).flatten_coefficient_modes().squeeze_modes(), method="uniform_1d",
  math_dtype=fp64), sender as input index, receiver as output index (atomic scatter inside). Node features MUST be ir_mul (MACE's cueq
  layout); feeding mul_ir gives wrong results (scaled err ~1) -- we wrap with cuet.TransposeIrrepsLayout in/out (exact copies, included in timing).
