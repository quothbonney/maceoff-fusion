#!/bin/bash
D=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/md
$D/job.sh prep.py water192 && $D/job.sh run_nve.py e3nn 0.5 200 10 smoke_e3nn && $D/job.sh run_nve.py flash 0.5 200 10 smoke_flash
