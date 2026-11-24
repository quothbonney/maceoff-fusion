#!/bin/bash
B=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/nodetp8_bwd
bash $B/job.sh test_local.py auto
bash $B/job.sh test_model.py /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/results/nodetp8_bwd/model_auto_final.json node,cueqsc_node auto
