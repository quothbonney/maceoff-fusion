"""MD validation: e3nn vs FP64-CG FlashTP. Common setup (import order matters: fp64cg flashTP first)."""
import os, sys, subprocess, time, json
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]
sys.path.insert(0, f"{R}/src/flashTP-fp64cg")
import flashTP_e3nn
assert flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-fp64cg"), flashTP_e3nn.__file__
sys.path.insert(0, f"{R}/scripts/flash_proto")
import flashtp_adapter as FA
from systems import get_system
import numpy as np, torch
from mace.calculators import MACECalculator

OUT = f"{R}/results/md"

def make_calc(backend):
    calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda",
                          default_dtype="float64", compute_stress=False)
    model = calc.models[0]; saved = None
    if backend == "flash":
        saved = FA.install(model); assert FA.is_installed(model) == [True, True]
    else:
        assert backend == "e3nn" and FA.is_installed(model) == [False, False]
    return calc, model, saved

def env_info():
    return dict(slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), flashTP_file=flashTP_e3nn.__file__,
                nvidia_smi=subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                                          capture_output=True, text=True).stdout.strip())
