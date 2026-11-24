#!/bin/bash
# supplementary final timing for wbox192 after the nl=auto policy change (cell list unsupported -> proto path)
R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study
export TIMING=1
bash $R/scripts/int9/job.sh bench9.py $R/results/int9/final_$1.json stock,hybrid+embed+node,nodehybrid+embed+node,gpustep,gpustep_hyb wbox192,aspirin
