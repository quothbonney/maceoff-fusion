#!/bin/bash
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
$R/venv-cueq/bin/python -I $R/scripts/validate7/prof7.py $R/results/validate7/$1 "$2" "$3" 2>&1 | grep -v "^DEBUG"
nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader
