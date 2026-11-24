#!/bin/bash
R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study
bash $R/scripts/prof8/job.sh flash_drv.py capture wbox648
bash $R/scripts/prof8/job.sh conv8.py $R/results/prof8/conv_j${SLURM_JOB_ID}.json aspirin,water192,org192,water648,org648,wbox648 flash,node,e3nn
