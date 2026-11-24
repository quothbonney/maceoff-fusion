#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/cudagraph6/build_graphcap.sh
# In-place build of the graph-capture FlashTP worktree (no pip install).
set -euo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export CUDA_ARCH_LIST=90 MAX_JOBS=8
export PATH=$R/venv-cueq/bin:$PATH
echo "job=${SLURM_JOB_ID:-} host=$(hostname) nvcc=$(which nvcc)"; nvcc --version | tail -1
cd "$R/src/flashTP-graphcap"; git log --oneline -1
time "$R/venv-cueq/bin/python" setup.py build_ext --inplace
ls -la flashTP_e3nn/*.so
