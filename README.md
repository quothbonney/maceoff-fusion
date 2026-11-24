# MACE-OFF23 medium (fp64) inference on H100 — study archive (Phases 1–9)

Methods, scripts, results and logs from a nine-phase study that took MACE-OFF23 medium fp64 energy+force(+stress)
evaluation on one H100 from stock MACE 0.3.17 to a GPU-resident, CUDA-graph-captured MD step, with fp64 parity
held to <= 1e-11 eV/atom and eV/A throughout.

Headline (complete step from GPU-resident coordinates, ms; Phase 9 job 1145/1146):

| system | atoms | stock MACE | final (GPUStep) | speedup |
|---|---|---|---|---|
| aspirin | 21 | 16.3 | 0.814 | 20x |
| water648 | 648 | 45.5 | 3.04 | 15x |
| wbox2187 (periodic) | 2187 | 208.7 | 10.74 | 19x |
| wbox5184 (periodic) | 5184 | 446 | 23.9 | 19x |

- `STUDY_INDEX.md` — per-phase index of every script and result (original study README).
- `scripts/`, `results/`, `logs/` — everything from the study under 5 MB per file (no `__pycache__`).
- `LARGE_FILES_MANIFEST.txt` — files >= 5 MB (captured tensors, ncu reports) kept on the cluster, with sha256.
- `patches/` — local FlashTP commits on top of upstream `d52e414` (fp64 CG fix; graph-capture stream fix). Not sent upstream.
- `env.sh` — environment recipe (contained CUDA 12.8.1, uv Python 3.11.17, venv-cueq).
- Not included: venvs, CUDA toolkit, upstream source checkouts, the MACE-OFF23 checkpoint
  (sha256 4842c52a…, from ACEsuit/mace-off @ 91a78c5; obtain from upstream under its own licence).

Paths in scripts assume the original root `R=.../maceoff-study`; set `R` to your checkout.
