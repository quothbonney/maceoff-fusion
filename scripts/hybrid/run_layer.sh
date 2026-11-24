#!/bin/bash
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"; nvidia-smi --query-gpu=index,name,uuid --format=csv
cd $R/scripts/hybrid
$R/venv-cueq/bin/python -I $R/scripts/hybrid/${1} $R/results/hybrid/${2}
echo rc=$?
