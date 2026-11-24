#!/bin/bash
# usage: job.sh <python script> [args...]  (one GPU job; logs header)
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) args=$*"; nvidia-smi --query-gpu=index,name,uuid --format=csv
S=$1; shift; exec $R/venv/bin/python -I $R/scripts/md/$S "$@"
