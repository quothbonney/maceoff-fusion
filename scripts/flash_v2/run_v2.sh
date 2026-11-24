#!/bin/bash
# usage: run_v2.sh <mode: core|extra|all>   (inside srun)
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"; nvidia-smi --query-gpu=index,name,uuid --format=csv
cd $R/scripts/flash_v2
MODES=${1:-core}; [ "$MODES" = all ] && MODES="core extra"
for m in $MODES; do for d in float32 float64; do
  echo "=== $m default=$d"
  $R/venv-cueq/bin/python -I $R/scripts/flash_v2/test_v2.py $m $d $R/results/flash_v2/test_${m}_default_${d}.json
  echo rc=$?
done; done
