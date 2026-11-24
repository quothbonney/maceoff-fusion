#!/bin/bash
# final timing: srun ... bash scripts/int9/run_final.sh <tag> [prof]   (each job.sh call takes TIMING_LOCK)
R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study
T=$1
export TIMING=1
bash $R/scripts/int9/job.sh bench9.py $R/results/int9/final_$T.json stock,hybrid+embed+node,nodehybrid+embed+node,gpustep,gpustep_hyb aspirin,water192,wbox192,org648,water648,wbox648,water2187,wbox2187,water5184,wbox5184
if [ "${2:-}" = prof ]; then
  bash $R/scripts/int9/job.sh prof9.py $R/results/int9/prof_final.json proto,gpustep wbox5184,water5184,wbox2187,water2187,wbox648,aspirin
fi
