#!/bin/bash
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=$R/venv/bin/python; O=$R/results/review; S=$R/scripts/review/mini_checks.py
echo "SLURM_JOB_ID=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name,uuid --format=csv,noheader | tee $O/gpu_$SLURM_JOB_ID.txt
F="shared_memory_bytes\|UserWarning\|_Jd, _W3j\|visitor"
$PY -I $S meta $R/venv/lib/python3.11/site-packages $O/kernel/meta_orig.pt 2>&1 | grep -v "$F"
$PY -I $S meta $R/src/flashTP-cgpatch $O/kernel/meta_cgpatch.pt 2>&1 | grep -v "$F"
CUDA_LAUNCH_BLOCKING=1 $PY -I $S empty $R/src/flashTP-cgpatch $O/empty_edges.json 2>&1 | grep -v "$F"
$PY -I $S refrep $R/src/flashTP-cgpatch $O/e3nn_ref_repeatability.json 2>&1 | grep -v "$F" | tail -2
