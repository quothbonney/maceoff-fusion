#!/bin/bash
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $R/tmp
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader | tee $R/results/precision/gpu_${SLURM_JOB_ID:-none}.txt
for v in ${ORDER:-orig fp64cg}; do
  if [ $v = orig ]; then root=-; else root=$R/src/flashTP-fp64cg; fi
  $R/venv/bin/python -I $R/scripts/precision/eval_precision.py $v $root $R/models/MACE-OFF23_medium.model \
     $R/tmp/precision_inputs.pt $R/results/precision/precision_$v${SUFFIX:-}.json 2>&1 | grep -v -i "warn\|torch.load\|_Jd, _W3j"
done
echo done $(date -Is)
