#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:25:00 bash $R/scripts/graph_e2e/run_e2e.sh <tag>
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $R/results/graph_e2e; echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
$R/venv-cueq/bin/python -I $R/scripts/graph_e2e/e2e.py $R/results/graph_e2e/e2e_$1.json
