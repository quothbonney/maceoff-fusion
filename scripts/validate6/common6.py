"""validate6 shared setup: same import order/env as scripts/cudagraph6/run_cg6.py (graphcap FlashTP, check_indices=False)."""
import os, sys, time, json, subprocess
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HB  # pins flashTP_e3nn to src/flashTP-graphcap
sys.path.insert(1, f"{R}/scripts/graph")
import numpy as np
import torch
import ase.io
from ase import Atoms
from mace.tools import torch_tools
from proto_graph import ProtoBuilder
from graphed_model import GraphedMACE, round_cap
assert HB.flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-graphcap/"), HB.flashTP_e3nn.__file__


def load(name):
    if name == "single_C":
        return Atoms("C", positions=[[0, 0, 0]])
    if name == "far_pair":
        return Atoms("OH", positions=[[0, 0, 0], [12.0, 0, 0]])
    return ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")


def eager(model, bd):
    with torch_tools.default_dtype("float64"):
        o = model(bd, compute_force=True, training=False)
    return o["energy"].detach().clone(), o["forces"].detach().clone()


def cmp(E, F, Er, Fr, n):
    d = (F - Fr).abs()
    return dict(dE_per_atom=float((E - Er).abs().max()) / max(n, 1), max_dF=float(d.max()) if d.numel() else 0.0,
                rms_dF=float(((F - Fr) ** 2).sum(-1).mean().sqrt()) if d.numel() else 0.0,
                finite=bool(torch.isfinite(E).all() and torch.isfinite(F).all()))


def gpuinfo():
    q = "index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def build_all(backs=("e3nn", "flash", "hybrid")):
    spec = dict(e3nn=("e3nn", "e3nn"), flash=("flashtp", "e3nn"), hybrid=("flashtp", "cueq"))
    calcs = {}
    for b in backs:
        calcs[b], _ = HB.build_calc(conv=spec[b][0], sc=spec[b][1], check_indices=False)
    return calcs, {k: ProtoBuilder(c) for k, c in calcs.items()}, {k: c.models[0] for k, c in calcs.items()}


def header():
    return dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), torch=torch.__version__,
                flashTP=HB.flashTP_e3nn.__file__)
