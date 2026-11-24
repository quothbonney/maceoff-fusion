#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 01:00:00 bash scripts/int9/job.sh <script.py> <args...>
# TIMING=1 -> acquire results/phase9/TIMING_LOCK (wait while another live job holds it), release at exit.
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
SP=$R/venv-cueq/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PATH=$R/venv-cueq/bin:$PATH
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"; nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.max.sm --format=csv,noheader
LOCK=$R/results/phase9/TIMING_LOCK
if [ "${TIMING:-0}" = 1 ]; then
  for i in $(seq 1 360); do
    if ( set -o noclobber; echo "$SLURM_JOB_ID int9 $(hostname) $(date -Is)" > $LOCK ) 2>/dev/null; then break; fi
    other=$(head -1 $LOCK | awk '{print $1}')
    if ! squeue -h -j "$other" 2>/dev/null | grep -q .; then echo "removing stale lock $(cat $LOCK)"; rm -f $LOCK; continue; fi
    echo "waiting for TIMING_LOCK held by $(cat $LOCK)"; sleep 10
  done
  trap 'if [ "$(head -1 $LOCK | awk "{print \$1}")" = "$SLURM_JOB_ID" ]; then rm -f $LOCK; fi' EXIT
  echo "TIMING_LOCK: $(cat $LOCK)"
  squeue -h -o "%i %j %u %N %T" | grep -v "^$SLURM_JOB_ID " | sed 's/^/other job: /'
fi
md5sum $R/scripts/int9/*.py | sed "s/^/md5 /"
S=$1; shift
$R/venv-cueq/bin/python -I $R/scripts/int9/$S "$@" 2>&1 | grep --line-buffered -v "^DEBUG" | grep --line-buffered -v Warning | grep --line-buffered -v "warnings.warn\|return visitor\|_Jd, _W3j\|torch.load(\|_warn_once"
nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader; echo "end=$(date -Is)"
