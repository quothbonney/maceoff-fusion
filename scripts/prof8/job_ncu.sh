#!/bin/bash
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PATH=$R/venv-cueq/bin:$PATH
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
PY="$R/venv-cueq/bin/python -I $R/scripts/prof8/flash_drv.py"
NCU=$R/toolchain/cuda-12.8.1-extract/nsight_compute/ncu
O=$R/results/prof8
$PY capture 2>&1 | grep -v Warn
$PY time 2>&1 | grep -v Warn
NITER=2 $NCU --set full --import-source no -k regex:sptp -f -o $O/flashtp_full $PY run 2>&1 | grep -v Warn | tail -30
$NCU -i $O/flashtp_full.ncu-rep --csv --page raw > $O/flashtp_full_raw.csv 2>&1; echo raw rc=$?
$NCU -i $O/flashtp_full.ncu-rep --page details --print-details all > $O/flashtp_full_details.txt 2>&1; echo det rc=$?
nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader
