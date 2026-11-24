#!/bin/bash
# Usage: srun -p h100 --gres=gpu:1 -c 8 -t 00:20:00 bash $R/scripts/graph/run_graph.sh <tag> [systems...]
set -u
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TAG=$1; shift
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} host=$(hostname) date=$(date -u +%FT%TZ) SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv
$R/venv/bin/python -I $R/scripts/graph/graph_profile.py $R/results/graph/graph_$TAG.json "$@"
echo rc=$?
echo END $(date -u +%FT%TZ)
