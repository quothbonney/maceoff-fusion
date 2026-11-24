#!/bin/bash
# srun ... bash job.sh <venv-name> <script.py> <args...>
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
V=$R/$1; shift
SP=$V/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PATH=$V/bin:$PATH
echo "job=$SLURM_JOB_ID host=$(hostname) venv=$V"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
S=$1; shift
$V/bin/python -I $R/scripts/oeq8/$S "$@" 2>&1 | grep -v "^DEBUG" | grep -v "warnings.warn\|return visitor"
nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader
