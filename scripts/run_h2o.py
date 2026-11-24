"""Baseline MACE-OFF23 medium inference on ASE H2O, one GPU. Usage: python -I run_h2o.py <model_path> <run_label>"""
import json, os, sys, time, warnings, platform
warnings.simplefilter("always")
caught = []
_orig = warnings.showwarning
def _rec(msg, cat, fn, ln, file=None, line=None):
    caught.append(f"{cat.__name__}: {msg} ({os.path.basename(fn)}:{ln})"); _orig(msg, cat, fn, ln, file, line)
warnings.showwarning = _rec

import numpy as np, torch, ase, mace
from ase.build import molecule
from mace.calculators import MACECalculator

model_path, label = sys.argv[1], sys.argv[2]
calc = MACECalculator(model_paths=model_path, device="cuda", compute_stress=False)
model = calc.models[0]
atoms = molecule("H2O")
atoms.calc = calc

def evaluate():
    atoms.calc.reset()
    torch.cuda.synchronize(); t = time.perf_counter()
    e = atoms.get_potential_energy(); f = atoms.get_forces()
    torch.cuda.synchronize(); return e, f, time.perf_counter() - t

e0, f0, t_first = evaluate()             # warmup
times, results = [], []
for _ in range(10):
    e, f, dt = evaluate(); times.append(dt); results.append((e, f.copy()))
e, f = results[-1]
assert np.isfinite(e) and np.all(np.isfinite(f)), "non-finite output"
max_de = max(abs(r[0] - e0) for r in results); max_df = max(np.abs(r[1] - f0).max() for r in results)

out = {
    "label": label, "host": platform.node(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0], "torch": torch.__version__,
    "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "mace": mace.__version__, "ase": ase.__version__,
    "model_path": model_path, "model_dtype": str(next(model.parameters()).dtype), "calc_default_dtype": str(torch.get_default_dtype()),
    "model_class": type(model).__name__, "r_max": float(model.r_max), "atoms": atoms.get_chemical_symbols(),
    "positions_A": atoms.positions.tolist(), "energy_eV": float(e), "forces_eV_per_A": f.tolist(),
    "first_call_s": t_first, "post_warmup_ms_median": 1e3*float(np.median(times)), "post_warmup_ms_min": 1e3*min(times),
    "within_run_max_abs_dE": float(max_de), "within_run_max_abs_dF": float(max_df), "warnings": caught,
}
print(f"[{label}] GPU={out['gpu']} dtype={out['model_dtype']} E = {e:.10f} eV")
for s, row in zip(out["atoms"], f): print(f"  {s}: Fx={row[0]: .10f} Fy={row[1]: .10f} Fz={row[2]: .10f} eV/A")
print(f"  post-warmup median {out['post_warmup_ms_median']:.2f} ms (n=10), first call {t_first:.2f} s, within-run max|dE|={max_de:.2e} max|dF|={max_df:.2e}")
json.dump(out, open(os.path.join(os.environ["R"], "results", f"h2o_{label}.json"), "w"), indent=2)
