#!/bin/bash
set -euxo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
$R/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11 -m venv $R/venv-luna
$R/venv-luna/bin/python -m pip install -r $R/results/pip_freeze_pre_flashtp_build.txt --extra-index-url https://download.pytorch.org/whl/cu128
# LunaEquivariance declares e3nn>=0.5.2,<0.7 but mace-torch 0.3.17 pins e3nn==0.4.4; install from pinned clone with --no-deps
$R/venv-luna/bin/python -m pip install --no-deps --no-build-isolation $R/src/lunatp
$R/venv-luna/bin/python -m pip freeze > $R/results/oeq8/pip_freeze_venv_luna.txt
$R/venv-luna/bin/python -I -c "import torch,e3nn;print(torch.__version__, e3nn.__version__)"
