#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:20:00 bash $R/scripts/flash_proto/run_parity.sh
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh; cd $R/results/integration
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
$R/venv/bin/python -I $R/scripts/flash_proto/parity_model.py $R/results/integration/model_parity.json H2O water192 water648 water1536
