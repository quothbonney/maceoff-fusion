#!/bin/bash
# job 2: method probe + fp64 repeat (run2) + fp32 run (explicitly labeled fp32; model is fp64, calculator casts)
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
nvidia-smi --query-gpu=index,name,uuid --format=csv
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib
PY="$R/venv-cueq/bin/python -I"; M=$R/models/MACE-OFF23_medium.model
for DT in float64 float32; do $PY $R/scripts/backend/cueq_methods_probe.py $M $DT $R/results/backend/cueq_methods_$DT.json 2>&1 | grep -v Warn | grep -v "warnings.warn\|torch.load"; done
$PY $R/scripts/backend/cueq_check_and_time.py $M float64 $R/results/backend/cueq_float64_run2.json 5 30
$PY $R/scripts/backend/cueq_check_and_time.py $M float32 $R/results/backend/cueq_float32_run1.json 5 30
echo rc=$?
