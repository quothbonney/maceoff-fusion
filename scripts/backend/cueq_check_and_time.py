"""cuEquivariance vs unmodified e3nn MACE: correctness + timing, same process, one GPU.
Usage: python -I cueq_check_and_time.py <model_path> <dtype float64|float32> <out_json> [n_warm] [n_time]
"""
import json, os, sys, time, platform, warnings, logging
warnings.simplefilter("always")
caught = []
_orig = warnings.showwarning
def _rec(msg, cat, fn, ln, file=None, line=None):
    caught.append(f"{cat.__name__}: {msg} ({os.path.basename(fn)}:{ln})"); _orig(msg, cat, fn, ln, file, line)
warnings.showwarning = _rec
logging.basicConfig(level=logging.INFO)

import numpy as np, torch, ase, mace
from ase.build import molecule
from ase import Atoms
from mace.calculators import MACECalculator
import cuequivariance, cuequivariance_torch
try:
    import cuequivariance_ops_torch; ops_ver = getattr(cuequivariance_ops_torch, "__version__", "?")
except Exception as e:
    ops_ver = f"IMPORT FAILED: {e!r}"

model_path, dtype, out_json = sys.argv[1:4]
n_warm = int(sys.argv[4]) if len(sys.argv) > 4 else 5
n_time = int(sys.argv[5]) if len(sys.argv) > 5 else 30
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def water_cluster(n_side=6, spacing=3.1, seed=0):  # identical to scripts/inspect_mace_layers.py
    rng = np.random.default_rng(seed)
    w = molecule("H2O"); base = w.positions - w.positions.mean(0)
    pos, sym = [], []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, dd = q
                Rm = np.array([[a*a+b*b-c*c-dd*dd, 2*(b*c-a*dd), 2*(b*dd+a*c)],
                               [2*(b*c+a*dd), a*a-b*b+c*c-dd*dd, 2*(c*dd-a*b)],
                               [2*(b*dd-a*c), 2*(c*dd+a*b), a*a-b*b-c*c+dd*dd]])
                pos.append(base @ Rm.T + spacing * np.array([i, j, k]) + rng.normal(scale=0.1, size=3))
                sym += w.get_chemical_symbols()
    return Atoms(symbols=sym, positions=np.concatenate(pos), pbc=False)

systems = {"H2O": molecule("H2O"), "water_cluster_6x6x6": water_cluster()}

def sync(): torch.cuda.synchronize()

def build(cueq):
    sync(); t = time.perf_counter()
    calc = MACECalculator(model_paths=model_path, device="cuda", default_dtype=dtype,
                          compute_stress=False, enable_cueq=cueq)
    sync(); return calc, time.perf_counter() - t

def evaluate(atoms):
    atoms.calc.reset()
    sync(); t = time.perf_counter()
    e = atoms.get_potential_energy(); f = atoms.get_forces()
    sync(); return float(e), f.copy(), time.perf_counter() - t

def n_edges(calc, atoms):
    from mace.data import config_from_atoms
    try:
        from mace.data.utils import config_from_atoms as c2
    except Exception:
        c2 = None
    from mace.tools import utils
    from mace.data import AtomicData
    cfg = config_from_atoms(atoms)
    d = AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
    return int(d.edge_index.shape[1])

out = {"host": platform.node(), "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
       "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0),
       "python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
       "mace": mace.__version__, "ase": ase.__version__, "cuequivariance": cuequivariance.__version__,
       "cuequivariance_torch": cuequivariance_torch.__version__, "cuequivariance_ops_torch": ops_ver,
       "model_path": model_path, "dtype": dtype, "n_warm": n_warm, "n_time": n_time,
       "timed_boundary": "atoms.calc.reset(); sync; [get_potential_energy(); get_forces()]; sync  (includes ASE->AtomicData neighbor list + H2D/D2H copy)",
       "setup": {}, "systems": {}}

calcs = {}
for name, cueq in (("e3nn", False), ("cueq", True)):
    log(f"building {name}")
    calc, t_build = build(cueq)
    m = calc.models[0]
    out["setup"][name] = {"construct_s": t_build, "param_dtype": str(next(m.parameters()).dtype),
                          "torch_default_dtype_after": str(torch.get_default_dtype()),
                          "module_types_sample": sorted({type(x).__module__ + "." + type(x).__name__ for x in m.modules()
                                                         if "cuequivariance" in type(x).__module__})[:20]}
    calcs[name] = calc
    log(f"{name}: construct {t_build:.2f}s, params {out['setup'][name]['param_dtype']}")

for sname, base_atoms in systems.items():
    res = {"n_atoms": len(base_atoms)}
    try: res["n_edges"] = n_edges(calcs["e3nn"], base_atoms)
    except Exception as e: res["n_edges"] = f"err {e!r}"
    runs = {}
    for name, calc in calcs.items():
        atoms = base_atoms.copy(); atoms.calc = calc
        e1, f1, t_first = evaluate(atoms)
        reps = [evaluate(atoms) for _ in range(3)]
        runs[name] = {"first_call_s": t_first, "E": [r[0] for r in reps], "F": [r[1] for r in reps]}
        rr_dE = max(abs(r[0] - reps[0][0]) for r in reps)
        rr_dF = max(float(np.abs(r[1] - reps[0][1]).max()) for r in reps)
        res[name] = {"first_call_s": t_first, "energy_eV": reps[0][0], "repeat_max_abs_dE_eV": rr_dE,
                     "repeat_max_abs_dF_eVA": rr_dF}
        log(f"{sname} {name}: E={reps[0][0]:.10f} first={t_first:.3f}s rep dE={rr_dE:.2e} dF={rr_dF:.2e}")
    Ee, Fe = runs["e3nn"]["E"][0], runs["e3nn"]["F"][0]
    Ec, Fc = runs["cueq"]["E"][0], runs["cueq"]["F"][0]
    dF = Fc - Fe
    res["cmp"] = {"abs_dE_eV": abs(Ec - Ee), "abs_dE_per_atom_eV": abs(Ec - Ee) / len(base_atoms),
                  "max_abs_dF_eVA": float(np.abs(dF).max()),
                  "rms_dF_eVA": float(np.sqrt((dF ** 2).mean())),
                  "max_norm_dF_eVA": float(np.linalg.norm(dF, axis=1).max()),
                  "max_abs_F_ref_eVA": float(np.abs(Fe).max())}
    log(f"{sname} cmp: {res['cmp']}")
    # timing
    for name, calc in calcs.items():
        atoms = base_atoms.copy(); atoms.calc = calc
        for _ in range(n_warm): evaluate(atoms)
        ts = np.array([evaluate(atoms)[2] for _ in range(n_time)]) * 1e3
        res[name]["timing_ms"] = {"median": float(np.median(ts)), "p10": float(np.percentile(ts, 10)),
                                  "p90": float(np.percentile(ts, 90)), "min": float(ts.min()),
                                  "mean": float(ts.mean()), "n": n_time, "all": ts.tolist()}
        log(f"{sname} {name}: median {np.median(ts):.2f} ms p10 {np.percentile(ts,10):.2f} p90 {np.percentile(ts,90):.2f} min {ts.min():.2f}")
    out["systems"][sname] = res

out["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
out["warnings"] = caught
json.dump(out, open(out_json, "w"), indent=2)
log("wrote", out_json)
