#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:15:00 bash $R/scripts/validate6/job_correct2.sh
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/validate6/env6.sh
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
$P -I $V/v_correct.py $O/correct_rerun.json check2b_recapture,check3_edge_cases 2>&1 | grep -v "^DEBUG\|warn\|Warning\|visitor\|torch.load"
