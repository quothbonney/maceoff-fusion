#!/bin/bash
# Run from login node: srun -p h100 --gres=gpu:1 -c 8 -t 00:15:00 bash $R/scripts/run_baseline.sh
set -euo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
cd "$R/results"   # cwd inside study dir (not next to model file)
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > gpu_info.csv
"$R/venv/bin/python" -I -c "import torch;print('cuda_available',torch.cuda.is_available(),torch.cuda.device_count())"
for i in 1 2; do "$R/venv/bin/python" -I "$R/scripts/run_h2o.py" "$R/models/MACE-OFF23_medium.model" run$i 2>&1 | tee "$R/logs/run$i.log"; done
"$R/venv/bin/python" -I - <<'PY'
import json,numpy as np,os
a,b=[json.load(open(f"{os.environ['R']}/results/h2o_run{i}.json")) for i in (1,2)]
dE=abs(a["energy_eV"]-b["energy_eV"]); dF=np.abs(np.array(a["forces_eV_per_A"])-np.array(b["forces_eV_per_A"])).max()
print(f"cross-run |dE|={dE:.3e} eV  max|dF|={dF:.3e} eV/A  bitwise_identical={dE==0 and dF==0}")
PY
