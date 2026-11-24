#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 16 -t 01:00:00 bash $R/scripts/build_flashtp.sh
set -euo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export CUDA_ARCH_LIST=90 MAX_JOBS=8
echo "host=$(hostname) CUDA_HOME=$CUDA_HOME nvcc=$(which nvcc)"; nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd "$R/src/flashTP"
time "$R/venv/bin/python" -m pip install . -v --no-build-isolation --no-deps
