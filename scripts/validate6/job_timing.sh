#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:25:00 bash $R/scripts/validate6/job_timing.sh TAG
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/validate6/env6.sh
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,clocks.mem --format=csv,noheader
$P -I $V/v_timing.py $O/timing_$1.json 2>&1 | grep -v "^DEBUG"
nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
