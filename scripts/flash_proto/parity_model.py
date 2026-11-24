"""Whole-model energy/force parity: unmodified MACE (e3nn) vs FlashTP adapter (both layers), fp64.
Usage: python -I parity_model.py <out_json> [systems...]"""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np, torch
from mace.calculators import MACECalculator
import flashtp_adapter as FA
from systems import get_system

R = os.environ["R"]; out_path = sys.argv[1]; names = sys.argv[2:] or ["H2O", "water192", "water648", "water1536"]
calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64", compute_stress=False)
model = calc.models[0]
NREP = 5

def run(atoms, n=NREP, perturb_seq=None):
    res = []
    for i in range(n):
        a = atoms.copy()
        if perturb_seq is not None: a.positions = perturb_seq[i]
        a.calc = calc; calc.reset()
        e = a.get_potential_energy(); f = a.get_forces().copy(); res.append((e, f))
    return res

def stats(ref, test, nat):
    dE = [abs(t[0] - r[0]) / nat for r, t in zip(ref, test)]
    dF = [np.abs(t[1] - r[1]) for r, t in zip(ref, test)]
    rms = [float(np.sqrt(((t[1] - r[1]) ** 2).mean())) for r, t in zip(ref, test)]
    return dict(max_abs_dE_per_atom_eV=float(max(dE)), max_abs_dF_eV_A=float(max(d.max() for d in dF)),
                max_rms_dF_eV_A=float(max(rms)), max_abs_F_ref=float(max(np.abs(r[1]).max() for r in ref)),
                rms_F_ref=float(np.sqrt(np.mean(ref[0][1] ** 2))))

def rep_var(res):
    return dict(max_abs_dE=float(max(abs(r[0] - res[0][0]) for r in res)),
                max_abs_dF=float(max(np.abs(r[1] - res[0][1]).max() for r in res)),
                bitwise_identical=all(r[0] == res[0][0] and np.array_equal(r[1], res[0][1]) for r in res))

out = dict(gpu=torch.cuda.get_device_name(0), slurm_job=os.environ.get("SLURM_JOB_ID"),
           gpu_uuid=subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           torch=torch.__version__, model_dtype=str(next(model.parameters()).dtype), nrep=NREP, systems={})
for name in names:
    atoms = get_system(name); nat = len(atoms)
    rng = np.random.default_rng(1); seq = [atoms.positions + rng.normal(scale=0.02, size=atoms.positions.shape) for _ in range(NREP)]
    ref = run(atoms); ref_seq = run(atoms, perturb_seq=seq)
    saved = FA.install(model); assert FA.is_installed(model) == [True, True]
    t0 = time.perf_counter(); fl = run(atoms); fl_seq = run(atoms, perturb_seq=seq)
    FA.uninstall(model, saved); assert FA.is_installed(model) == [False, False]
    back = run(atoms, n=1)
    d = dict(n_atoms=nat, E_ref=ref[0][0], E_flash=fl[0][0],
             fixed_geometry=stats(ref, fl, nat), perturbed_sequence=stats(ref_seq, fl_seq, nat),
             ref_repeat=rep_var(ref), flash_repeat=rep_var(fl),
             reverted_bitwise_equal_ref=bool(back[0][0] == ref[0][0] and np.array_equal(back[0][1], ref[0][1])),
             all_finite=bool(all(np.isfinite(e) and np.isfinite(f).all() for e, f in fl + fl_seq)))
    out["systems"][name] = d
    print(name, json.dumps(d, indent=1, default=float), flush=True)
json.dump(out, open(out_path, "w"), indent=2, default=float)
