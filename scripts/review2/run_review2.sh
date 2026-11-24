#!/bin/bash
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"; nvidia-smi --query-gpu=index,name,uuid --format=csv
cd $R/scripts/review2
O=$R/results/review2
$R/venv-cueq/bin/python -I $R/scripts/review2/review2.py float64 $O/review2_default_float64 ${1:-}; echo rc=$?
$R/venv-cueq/bin/python -I $R/scripts/review2/review2.py float32 $O/review2_default_float32 ${2:-parity_fp64,cg_tables,fp32,graph_changes,zero_edge,input_checks,build_errors,cleanup}; echo rc=$?
$R/venv-cueq/bin/python -I $R/scripts/review2/compare_defaults.py $O/review2_default_float32.npz $O/review2_default_float64.npz $O/compare_defaults.json; echo rc=$?
