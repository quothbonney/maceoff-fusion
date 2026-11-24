#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:25:00 bash $R/scripts/validate6/job_main.sh
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/scripts/validate6/env6.sh
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,clocks.mem --format=csv,noheader
# check 4: parity (separate processes per build)
PD=$R/tmp/validate6_parity_$SLURM_JOB_ID; mkdir -p $PD
$P -I $V/v_parity.py run graphcap $PD 2>&1 | grep -v "^DEBUG" | tail -3
$P -I $V/v_parity.py run cgpatch $PD 2>&1 | grep -v "^DEBUG" | tail -3
$P -I $V/v_parity.py compare $PD $O/parity_graphcap_vs_cgpatch.json 2>&1 | tail -40
export PYTHONPATH=$R/src/flashTP-graphcap
T="$R/src/flashTP-graphcap/example/flashTP_test.py"; C="$R/src/flashTP-graphcap/ir_config/mace/medium.json"
for cfg in "0 4096" "1 4096" "1 32768"; do set -- $cfg
  $P $V/upstream_check.py "$T" "$C" $1 $2 fp64 False 1 2>&1 | grep -E "SUMMARY|Error|error" | tail -3
done
unset PYTHONPATH
# checks 1-3
$P -I $V/v_correct.py $O/correct.json 2>&1 | grep -v "^DEBUG"
# check 5 (timing run A)
$P -I $V/v_timing.py $O/timing_A.json 2>&1 | grep -v "^DEBUG"
nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
