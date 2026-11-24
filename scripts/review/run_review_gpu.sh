#!/bin/bash
# GPU review job: kernel parity (orig / fp64cg / cgpatch), upstream example on cgpatch, adapter tests.
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=$R/venv/bin/python; OUT=$R/results/review; L=$R/logs/review
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader | tee $OUT/gpu_$SLURM_JOB_ID.txt
F="shared_memory_bytes"
for t in "$R/venv/lib/python3.11/site-packages orig" "$R/src/flashTP-fp64cg fp64cg" "$R/src/flashTP-cgpatch cgpatch"; do
  set -- $t
  $PY -I $R/scripts/review/kernel_parity.py $1 $2 $OUT/kernel 2>&1 | grep -v "$F\|UserWarning\|_Jd, _W3j" > $L/kernel_parity_$2.log
  echo "kernel_parity $2 exit=${PIPESTATUS[0]}"
done
for dt in fp64 fp32; do for cfg in mace/medium.json:0 mace/medium.json:1 sevennet/sevennet-l3i5.json:1; do
  f=${cfg%%:*}; li=${cfg##*:}
  $PY -I -c "
import sys, runpy; T='$R/src/flashTP-cgpatch'; sys.path.insert(0, T)
import flashTP_e3nn; assert flashTP_e3nn.__file__.startswith(T), flashTP_e3nn.__file__
sys.argv=['flashTP_test.py', T+'/ir_config/$f', '$li', '32768', '$dt', 'False', '1']
runpy.run_path(T+'/example/flashTP_test.py', run_name='__main__')" 2>&1 | grep -v "$F\|UserWarning\|_Jd, _W3j\|DEBUG\|INFO" > $L/example_cgpatch_${dt}_${f//\//_}_L$li.log
  echo "example $dt $cfg exit=${PIPESTATUS[0]} $(grep -c 'All Good' $L/example_cgpatch_${dt}_${f//\//_}_L$li.log)/7 All Good"
done; done
$PY -I $R/scripts/review/adapter_review.py $R/src/flashTP-cgpatch $OUT/adapter_review.json 2>&1 | grep -v "$F\|UserWarning\|_Jd, _W3j" > $L/adapter_review.log
echo "adapter_review exit=${PIPESTATUS[0]}"
