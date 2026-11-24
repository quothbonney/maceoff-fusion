#!/bin/bash
# srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash $R/scripts/run_flashtp_checks.sh
set -uo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
# e3nn 0.4.4 loads its bundled constants.pt with torch.load default weights_only=True (torch>=2.6) and fails;
# mace/__init__.py:5 sets this same variable in-process. Set only for these test processes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$R/results"; P="$R/venv/bin/python"; T="$R/src/flashTP/example/flashTP_test.py"; C="$R/src/flashTP/ir_config/mace/medium.json"
"$P" -I -c "import flashTP_e3nn,torch;print('import ok',flashTP_e3nn.__file__,torch.cuda.get_device_name(0))"
for cfg in "0 4096" "1 4096" "0 32768" "1 32768" "1 131072"; do set -- $cfg
  SECONDS=0; echo "===== medium.json layer=$1 edges=$2 fp64"
  "$P" -I "$R/scripts/flashtp_fp64_check.py" "$T" "$C" $1 $2 fp64 False 1 2>&1 | grep -vE "^DEBUG"; echo "wall ${SECONDS}s"
done
