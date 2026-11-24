#!/bin/bash
# Usage: srun -p h100 --gres=gpu:1 -c 8 -t 00:25:00 bash run_perf.sh <tag> [systems...]
set -u
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
TAG=$1; shift
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} host=$(hostname) date=$(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
P=$R/venv/bin/python
if [ "${SKIP_BENCH:-0}" != 1 ]; then $P -I $R/scripts/perf/bench.py $R/results/perf/bench_$TAG.json "$@"; fi
if [ "${SKIP_PROF:-0}" != 1 ]; then $P -I $R/scripts/perf/prof.py $R/results/perf/prof_$TAG.json $R/logs/perf/$TAG; fi
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv
echo END $(date -u +%FT%TZ)
