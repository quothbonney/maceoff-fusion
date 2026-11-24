#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/graphed7/job_test.sh
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
$R/venv-cueq/bin/python -I $R/scripts/graphed7/${1:-test_graphed.py} ${2:-$R/results/graphed7/tests.json} ${3:-} 2>&1 | grep -v "^DEBUG"
