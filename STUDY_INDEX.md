# maceoff-study — MACE-OFF23 medium baseline (2026-11-08)
Root: /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study (moved from /mnt/home/jordan+123/maceoff-study on 2026-11-08; venv rebuilt, freeze identical)
Env: `source env.sh` (all caches/TMPDIR here, CUDA_VISIBLE_DEVICES=0)
Run: `srun -p h100 --gres=gpu:1 -c 8 -t 00:15:00 bash scripts/run_baseline.sh` (from slurm-login-0)
Layout: python/ (uv-managed CPython 3.11.17), venv/, models/, scripts/, results/, logs/, cache/, tmp/, src/ (empty, reserved)
Checkpoint sha256 4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7, 18350596 bytes (mace-off HEAD 91a78c5)

## Phase 2 (2026-11-08): FlashTP build + FP64 checks + MACE layer inventory
- src/flashTP @ d52e41454afc9636cd51757b65df44e7bcdca178 ; src/mace-v0.3.17 @ e37c8583a8b1b586e5888fa6dca86e8f591e2606 (reference only, not installed)
- CUDA toolkit: NVIDIA cuda_12.8.1_570.124.06_linux.run (md5 verified) extracted (--extract, no installer) to toolchain/cuda-12.8.1-extract, composed (hardlinks) into toolchain/cuda-12.8.1 (nvcc V12.8.93). env.sh sets CUDA_HOME/PATH. System has only CUDA 13.0.
- Build: scripts/build_flashtp.sh (CUDA_ARCH_LIST=90 MAX_JOBS=8, pip --no-build-isolation --no-deps) -> logs/17_flashtp_build.log; sm_90 SASS (results/flashtp_sass_archs.txt)
- Venv changes: +ninja, +wheel (build tools, --no-deps), +flashTP_e3nn 0.1.0. See results/pip_freeze_{phase2_start,pre_flashtp_build,post_flashtp_build}.txt
- FP64 checks: scripts/run_flashtp_checks.sh + scripts/flashtp_fp64_check.py (wraps upstream example unmodified) -> logs/22_flashtp_fp64_checks.log, results/flashtp_check_*.json
- Baselines: results/baseline_phase2_pre_build/, logs/12_*, logs/21_* (post-build) — identical to phase 1
- Inventory: scripts/inspect_mace_layers.py -> results/mace_layer_inventory.json, logs/20_inspect_layers.log

## Phase 3 (2026-11-08): FlashTP integration study
- Prototype adapter (in-memory, reversible; no MACE edits): scripts/flash_proto/flashtp_adapter.py (install/uninstall); systems.py; parity_model.py (+_fp64cg); run_parity*.sh
- Whole-model parity: results/integration/model_parity.json (unmodified FlashTP), model_parity_fp64cg.json (exp branch); model_parity_BUGGY_edgecache.json = superseded run exposing an adapter edge-index cache bug (fixed)
- Precision agent: scripts/precision/, results/precision/ (fp64cg.diff), src/flashTP-fp64cg (worktree, branch exp/fp64-cg @ 02de84c9)
- Performance agent: scripts/perf/, results/perf/bench_main.json, prof_main.json
- Backend agent (cuEquivariance 0.10.0, isolated venv-cueq): scripts/backend/, results/backend/

## Phase 4 (2026-11-09): hybrid FlashTP-FP64CG conv + cueq symmetric contraction; MD; review; 4-backend timing
- Hybrid: scripts/hybrid/hybrid_backend.py (build_calc, describe), results/hybrid/{layer,model}_parity.json  [runs in venv-cueq]
- MD/NVE: scripts/md/, results/md/PROTOCOL.md, summary.json
- Timing: scripts/perf2/bench4.py, results/perf2/bench4_run{1,2}.json
- Review + minimal upstream CG patch: src/flashTP-cgpatch (branch upstream/fp64-cg-minimal @ 0e4d820, NOT pushed), results/review/{0001-*.patch,PATCH_NOTES.md,adapter_review.json}
- Preservation check: results/phase4/preserve_manifest_start.txt (all prior files verified unchanged at end)

## Phase 5 (2026-11-09): hardened v2 backend, organic/periodic workloads, graph-build profiling + GPU graph prototype, review
- v2 adapter/hybrid (FlashTP from src/flashTP-cgpatch @ 0e4d820): scripts/flash_v2/ (READY, LIMITATIONS.md), results/flash_v2/
- Workloads (aspirin, A-T pair, organic clusters 196/385/650, water series 192-648, periodic water boxes): scripts/workloads/, results/workloads/summary_v2.md
- Graph-build decomposition + GPU graph builder prototype: scripts/graph/, results/graph/; end-to-end: scripts/graph_e2e/, results/graph_e2e/e2e_run{1,2}.json
- Independent review: results/review2/REVIEW.md
- Preservation: results/phase5/preserve_manifest_start.txt verified at end; hashes in results/phase5/source_hashes.txt

## Phase 6 (2026-11-09): profiling, fusion-boundary analysis, CUDA-graph capture experiment
- Profiling (fixed graph, 3 backends x 5 systems): scripts/profile6/, results/profile6/SUMMARY.md
- Fusion design / ranked opportunities: results/design6/DESIGN.md
- CUDA-graph experiment: scripts/cudagraph6/graphed_model.py; FlashTP branch exp/graph-capture @ 415e7a3 (src/flashTP-graphcap, in-place build); results/cudagraph6/
- Independent validation: results/validate6/VALIDATION.md
- Preservation: results/phase6/preserve_manifest_start.txt verified; hashes results/phase6/source_hashes.txt

## Phase 7 (2026-11-10): corrected graphed runner + fused kernels
- Contract: results/phase7/INTERFACES.md
- A corrected graphed runner (keyed graphs, recapture, stress): scripts/graphed7/, results/graphed7/READY
- B fused edge embedding (CUDA load_inline: SH l<=3 + Bessel*poly cutoff, fwd+bwd): scripts/fuse7_embed/, results/fuse7_embed/READY
- C fused post-conv node op (Triton fp64: L0 linear+1/avg+skip_tp per-species fold; L1 linear+1/avg; L1 skip_tp): scripts/fuse7_node/, results/fuse7_node/READY
- D profile + integrated gate + timing: results/validate7/{PROFILE,VALIDATION}.md
- Preservation: results/phase7/preserve_manifest_start.txt verified; hashes results/phase7/source_hashes.txt

## Phase 8 (2026-11-10): node-centric FP64 TP convolution
- Contract: results/phase8/INTERFACES.md
- A existing backends (LunaTP @0ead67d in src/lunatp + venv-luna; OEQ 0.8.0 in venv-oeq; cuEq fused): results/oeq8/{BASELINES.md,LUNA_READY,SOURCES.md}
- B node-centric forward (CUDA load_inline): scripts/nodetp8_fwd/, results/nodetp8_fwd/READY
- C CSR + src-centric fused backward + autograd/install: scripts/nodetp8_bwd/, results/nodetp8_bwd/READY
- D Nsight Compute FlashTP profile + integration ("node-hybrid"): results/prof8/{FLASHTP_NCU.md,INTEGRATION.md}
- Preservation: results/phase8/preserve_manifest_start.txt verified; hashes results/phase8/source_hashes.txt

## Phase 9 (2026-11-10): GPU-resident MD step
- Contract: results/phase9/INTERFACES.md
- A GPU cell-list neighbour builder (dst-sorted edges + CSR, no sort/sync, overflow flags): scripts/nl9/cell_nl.py, results/nl9/READY
- C integration: scripts/int9/{gpu_step.py,nodeconv_ext.py} (whole step = NL + model fwd/bwd in one CUDA graph), results/int9/{INTEGRATION.md,final_tables.md,prof_tables.md,correctness_table.md}
- B GPU velocity-Verlet + dynamic correctness + NVE drift + B3 throughput: scripts/md9/, results/md9/MD9.md
- Preservation: results/phase9/preserve_manifest_start.txt verified (482 files); hashes results/phase9/source_hashes.txt
