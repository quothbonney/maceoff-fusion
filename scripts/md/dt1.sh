#!/bin/bash
D=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/md
$D/job.sh run_nve.py e3nn 1.0 2000 5 e3nn_dt1.0 && $D/job.sh run_nve.py flash 1.0 2000 5 flash_dt1.0
