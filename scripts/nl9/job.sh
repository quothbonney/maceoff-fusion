#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 01:00:00 bash job.sh <script.py> <args...>
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PATH=$R/venv-cueq/bin:$PATH
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
S=$1; shift
$R/venv-cueq/bin/python -I $R/scripts/nl9/$S "$@" 2>&1 | grep -v "^DEBUG" | grep -v "Warning\|warnings.warn\|return visitor\|_Jd, _W3j\|torch.load(\|_warn_once"
nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader
