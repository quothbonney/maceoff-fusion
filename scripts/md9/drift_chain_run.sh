#!/bin/bash
# usage: drift_chain.sh <provider> [tag suffix]   (runs the drift set sequentially in one GPU job; MD9_NL env for gpustep)
P=$1; X=${2:-}; D=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/md9
bash $D/job.sh drift9.py $P water192 0.5 4000 10 ${P}${X}_water192_dt0.5
bash $D/job.sh drift9.py $P wbox648 0.5 4000 20 ${P}${X}_wbox648_dt0.5
bash $D/job.sh drift9.py $P wbox648 0.25 8000 40 ${P}${X}_wbox648_dt0.25
