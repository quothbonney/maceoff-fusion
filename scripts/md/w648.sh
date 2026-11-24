#!/bin/bash
D=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/md
$D/job.sh prep.py water648 && $D/job.sh run_nve.py e3nn 0.5 3000 10 w648_e3nn_dt0.5 water648 && $D/job.sh run_nve.py flash 0.5 3000 10 w648_flash_dt0.5 water648
