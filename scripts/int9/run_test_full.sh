#!/bin/bash
# correctness (not timing): bash scripts/int9/job.sh is called per process to keep each in a fresh CUDA context
R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study
bash $R/scripts/int9/job.sh test9.py $R/results/int9/test_full_a.json auto aspirin,water192,wbox192,org648,water648,wbox648
for s in water2187 wbox2187 water5184 wbox5184; do
  bash $R/scripts/int9/job.sh test9.py $R/results/int9/test_full_$s.json auto $s sysonly
done
