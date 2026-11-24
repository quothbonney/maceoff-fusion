#!/bin/bash
# Usage: srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/profile6/run6.sh <tag> <backends> <systems...>
set -u
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} host=$(hostname) date=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,clocks.mem --format=csv
TAG=$1; BE=$2; shift 2
$R/venv-cueq/bin/python -I $R/scripts/profile6/prof6.py $R/results/profile6/raw_$TAG.json "$BE" "$@"; echo rc=$?
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv
echo END $(date -u +%FT%TZ)
