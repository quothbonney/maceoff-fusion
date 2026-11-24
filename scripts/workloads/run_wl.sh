#!/bin/bash
# Usage: srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/workloads/run_wl.sh <tag> <backends> <systems...>
#   env passthrough: MODE (corr|full), BACKEND_SET (v1|v2), RELAX=1 (build+prerelax systems first, then exit)
set -u
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} host=$(hostname) date=$(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
squeue -h -o "%i %u %j %b %N" 2>/dev/null
cd $R/scripts/workloads
if [ "${RELAX:-0}" = 1 ]; then
  $R/venv-cueq/bin/python -I $R/scripts/workloads/systems2.py build --relax; echo rc=$?
else
  TAG=$1; BE=$2; shift 2
  $R/venv-cueq/bin/python -I $R/scripts/workloads/bench_wl.py $R/results/workloads/wl_$TAG.json "$BE" "$@"; echo rc=$?
fi
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv
echo END $(date -u +%FT%TZ)
