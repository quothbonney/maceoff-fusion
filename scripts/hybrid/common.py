import os, sys, subprocess
R = os.environ["R"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_backend as HB  # noqa (does the fp64cg sys.path insert + asserts)
sys.path.insert(1, f"{R}/scripts/flash_proto")
from systems import get_system  # noqa
import torch, numpy as np  # noqa
MODEL = f"{R}/models/MACE-OFF23_medium.model"

def env_info():
    import mace, e3nn, cuequivariance as cue
    return dict(slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                torch=torch.__version__, mace=mace.__version__, e3nn=e3nn.__version__, cuequivariance=cue.__version__,
                flashTP_e3nn=HB.flashTP_e3nn.__file__)

def metr(t, r):
    t = t.detach().double(); r = r.detach().double(); d = (t - r).abs()
    rr = r.abs().max().item(); rms_r = r.pow(2).mean().sqrt().item()
    return dict(max_abs=d.max().item(), max_abs_rel=d.max().item() / rr if rr else None,
                rms_rel=(d.pow(2).mean().sqrt().item() / rms_r) if rms_r else None, max_abs_ref=rr, shape=list(r.shape))
