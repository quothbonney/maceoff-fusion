#!/bin/bash
# Usage (from login node): srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/backend/run_cueq.sh <dtype> <tag>
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
DT=${1:-float64}; TAG=${2:-run1}
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) SLURM_STEP_GPUS=$SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid --format=csv
nvidia-smi -L
SP=$R/venv-cueq/lib/python3.11/site-packages
echo "--- probe: import cuequivariance_ops_torch WITHOUT LD_LIBRARY_PATH"
$R/venv-cueq/bin/python -I -c "import torch, cuequivariance_ops_torch as o; print('ops import ok', o.__version__)" 2>&1 | tail -1
# libcue_ops.so links libnvrtc.so.12/libcublas.so.12 by soname; torch 2.11 does not preload nvrtc, so expose the pip nvidia libs
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "--- probe: WITH LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
$R/venv-cueq/bin/python -I -c "import torch, cuequivariance_ops_torch as o; print('ops import ok', o.__version__)" 2>&1 | tail -1
$R/venv-cueq/bin/python -I $R/scripts/backend/cueq_check_and_time.py $R/models/MACE-OFF23_medium.model $DT $R/results/backend/cueq_${DT}_${TAG}.json 5 30
echo rc=$?
