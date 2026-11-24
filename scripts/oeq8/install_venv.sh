#!/bin/bash
set -euxo pipefail
source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh
$R/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11 -m venv $R/venv-oeq
$R/venv-oeq/bin/python -m pip install -r $R/results/pip_freeze_pre_flashtp_build.txt --extra-index-url https://download.pytorch.org/whl/cu128
$R/venv-oeq/bin/python -m pip install openequivariance==0.8.0 jinja2 pybind11 ninja setuptools -c <(grep -i "^torch==\|^numpy==" $R/results/pip_freeze_pre_flashtp_build.txt)
$R/venv-oeq/bin/python -m pip freeze > $R/results/oeq8/pip_freeze_venv_oeq.txt
$R/venv-oeq/bin/python -I -c "import torch;print(torch.__version__)"
