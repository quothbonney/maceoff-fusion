#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/cudagraph6/run_all.sh <tag> [systems] [backends]
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=$R/venv-cueq/bin/python; D=$R/scripts/cudagraph6; TAG=${1:-run}; shift || true; ARGS=("$@")
echo "job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
if [ "${FEAS:-1}" = 1 ]; then
  for b in e3nn flash hybrid; do echo "== stock feas $b"; timeout 300 $P -I $D/feas_stock.py $b 2>&1 | grep -v "^DEBUG" | tail -8; done
fi
if [ "${CHECKS:-1}" = 1 ]; then
  export PYTHONPATH=$R/src/flashTP-graphcap
  T="$R/src/flashTP-graphcap/example/flashTP_test.py"; C="$R/src/flashTP-graphcap/ir_config/mace/medium.json"
  $P -c "import flashTP_e3nn;print('flashTP from',flashTP_e3nn.__file__)"
  for cfg in "0 4096" "1 4096" "1 32768"; do set -- $cfg
    $P $D/flashtp_fp64_check_gc.py "$T" "$C" $1 $2 fp64 False 1 2>&1 | grep -E "SUMMARY|Error|error" | tail -3
  done
  unset PYTHONPATH
fi

$P -I $D/run_cg6.py $R/results/cudagraph6/cg6_${TAG}.json "${ARGS[@]}" 2>&1 | grep -v "^DEBUG"
nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader
