#!/bin/bash
# usage: srun ... bash timing_job.sh <script.py> [args]  -- acquires results/phase9/TIMING_LOCK (shared with agent C)
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
L=$R/results/phase9/TIMING_LOCK
until ( set -o noclobber; echo "$SLURM_JOB_ID md9 $(date -Is)" > $L ) 2>/dev/null; do
  echo "waiting for TIMING_LOCK: $(cat $L 2>/dev/null)"; sleep 20; done
trap 'grep -q "^$SLURM_JOB_ID md9" $L && rm -f $L' EXIT
echo "acquired TIMING_LOCK $(date -Is)"; squeue -h -o "%i %j %u %b %T" 2>/dev/null
bash $R/scripts/md9/job.sh "$@"
