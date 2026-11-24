"""MACE-OFF23 medium fp64: e3nn vs FlashTP(FP64-CG) vs cuEquivariance [vs hybrid], ONE process, interleaved blocks.
Usage: python -I bench4.py <out_json> <backends comma-list> [systems...]
Boundaries identical to scripts/perf/bench.py (graph_cpu, graph_total, fwd, bwd, fwd_bwd, ase_total).
Each backend is its own MACECalculator (separate model object), default_dtype=float64, compute_stress=False.
"""
import os, sys, json, time, subprocess
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]
sys.path.insert(0, f"{R}/src/flashTP-fp64cg")          # FP64-CG FlashTP, must precede any flashTP_e3nn import
import flashTP_e3nn
assert flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-fp64cg/"), flashTP_e3nn.__file__
import flashTP_e3nn.flashTP as _ftp
assert "build_unique_cg" in open(_ftp.__file__).read(), "not the FP64-CG flashTP"
sys.path.insert(0, f"{R}/scripts/flash_proto")
import flashtp_adapter as FA
from systems import get_system
import numpy as np, torch
from mace.calculators import MACECalculator
from mace import data as mace_data
from mace.tools import torch_tools

out_path = sys.argv[1]
BACKENDS = sys.argv[2].split(",")
names = sys.argv[3:] or ["H2O", "water648", "water1536"]
WARMUP = int(os.environ.get("WARMUP", 10)); BLOCK = int(os.environ.get("BLOCK", 10))
NBLOCKS = {"H2O": 6, "water192": 6, "water648": 6, "water1536": 4}
MODEL = f"{R}/models/MACE-OFF23_medium.model"
PARITY_E_PER_ATOM = 1e-12; PARITY_F = 1e-12      # expected fp64 level
PARITY_HARD = 1e-9                                 # above this: backend invalid
sync = torch.cuda.synchronize
BNAMES = ["graph_cpu", "graph_total", "fwd", "bwd", "fwd_bwd", "ase_total"]


def nvsmi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


out = dict(slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=torch.cuda.get_device_name(0),
           nvidia_smi_start=nvsmi(), torch=torch.__version__, flashTP_e3nn=flashTP_e3nn.__file__,
           flashtp_so=[os.path.realpath(os.path.join(os.path.dirname(flashTP_e3nn.__file__), f))
                       for f in os.listdir(os.path.dirname(flashTP_e3nn.__file__)) if f.endswith(".so")],
           backends=BACKENDS, warmup=WARMUP, block=BLOCK, nblocks=NBLOCKS, setup={}, systems={})
import mace, cuequivariance
out["mace"] = mace.__version__; out["cuequivariance"] = cuequivariance.__version__


def tsec(fn):
    sync(); t0 = time.perf_counter(); r = fn(); sync(); return r, (time.perf_counter() - t0) * 1e3


def build(be):
    info = {}
    if be == "e3nn":
        calc, info["construct_ms"] = tsec(lambda: MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False))
    elif be == "flash64":
        calc, info["construct_ms"] = tsec(lambda: MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False))
        _, info["FA_install_ms"] = tsec(lambda: FA.install(calc.models[0]))
        assert all(FA.is_installed(calc.models[0]))
        assert type(calc.models[0].interactions[0].conv_tp.ftp).__module__.startswith("flashTP_e3nn")
        info["ftp_module_file"] = sys.modules[type(calc.models[0].interactions[0].conv_tp.ftp).__module__].__file__
        sys.path.insert(0, f"{R}/scripts/hybrid"); import hybrid_backend as HB   # reuse its generic CG-table check
        info["flash_cg_dtypes"] = HB.flash_cg_dtypes(calc.models[0])
        assert all("torch.float32" not in v["tensor_dtypes"] and v["cg_not_fp32_representable"] > 0 for v in info["flash_cg_dtypes"].values()), info["flash_cg_dtypes"]
    elif be == "cueq":
        calc, info["construct_ms"] = tsec(lambda: MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False, enable_cueq=True))
    elif be == "hybrid":
        sys.path.insert(0, f"{R}/scripts/hybrid")
        import hybrid_backend
        (calc, hinfo), info["construct_ms"] = tsec(lambda: hybrid_backend.build_calc(MODEL, "cuda"))
        info["hybrid_info"] = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else repr(v)) for k, v in dict(hinfo or {}).items()}
        info["hybrid_backend_file"] = hybrid_backend.__file__
    else:
        raise KeyError(be)
    m = calc.models[0]
    info["param_dtype"] = str(next(m.parameters()).dtype)
    info["module_types"] = sorted({type(x).__module__ + "." + type(x).__name__ for x in m.modules()
                                   if any(s in type(x).__module__ for s in ("cuequivariance", "flash", "hybrid"))})
    return calc, info


calcs = {}
for be in BACKENDS:
    calcs[be], out["setup"][be] = build(be)
# first/second ASE call on water648 per backend (first call includes lazy init / JIT)
w648 = get_system("water648")
for be in BACKENDS:
    a = w648.copy(); a.calc = calcs[be]
    for k in ("first", "second", "third"):
        _, out["setup"][be][f"{k}_ase_call_water648_ms"] = tsec(lambda: (calcs[be].reset(), a.get_potential_energy(), a.get_forces()))
print("SETUP", json.dumps({be: {k: v for k, v in s.items() if k.endswith("_ms")} for be, s in out["setup"].items()}), flush=True)


def make_batch_dict(calc, atoms):
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v): b[k] = v.to(dtype=torch.float64)
    return b.to_dict()


def graph_cpu(calc, atoms):
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    with torch_tools.default_dtype("float64"):
        cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
        return mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)


def timed(fn, pre=None):
    if pre is not None: pre()
    sync(); t0 = time.perf_counter(); fn(); sync(); return (time.perf_counter() - t0) * 1e3


def boundaries(calc, atoms, bd):
    model = calc.models[0]
    def fwd():
        with torch_tools.default_dtype("float64"): return model(bd, compute_force=False, training=False)
    state = {}
    def pre_bwd(): state["out"] = fwd()
    def bwd():
        torch.autograd.grad([state["out"]["energy"]], [bd["positions"]], grad_outputs=[torch.ones_like(state["out"]["energy"])])
    def fwd_bwd():
        with torch_tools.default_dtype("float64"): return model(bd, compute_force=True, training=False)
    a = atoms.copy(); a.calc = calc
    def ase_total():
        calc.reset(); a.get_potential_energy(); a.get_forces()
    return dict(graph_cpu=(lambda: graph_cpu(calc, atoms), None), graph_total=(lambda: make_batch_dict(calc, atoms), None),
                fwd=(fwd, None), bwd=(bwd, pre_bwd), fwd_bwd=(fwd_bwd, None), ase_total=(ase_total, None))


def stats(x):
    x = np.asarray(x); p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=int(x.size), median=float(p50), p10=float(p10), p90=float(p90), min=float(x.min()),
                iqr=float(p75 - p25), mean=float(x.mean()))


for name in names:
    atoms = get_system(name); nat = len(atoms)
    bds = {be: make_batch_dict(calcs[be], atoms) for be in BACKENDS}
    nedge = int(bds["e3nn"]["edge_index"].shape[1])
    Bs = {be: boundaries(calcs[be], atoms, bds[be]) for be in BACKENDS}
    times = {be: {b: [] for b in BNAMES} for be in BACKENDS}
    ref, mem, first = {}, {}, {}
    for be in BACKENDS:
        a = atoms.copy(); a.calc = calcs[be]
        _, first[be] = tsec(lambda: (calcs[be].reset(), a.get_potential_energy(), a.get_forces()))
        for _ in range(WARMUP):
            for bn, (fn, pre) in Bs[be].items(): timed(fn, pre)
        calcs[be].reset(); sync()
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        e = a.get_potential_energy(); f = a.get_forces().copy(); sync()
        mem[be] = dict(peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20, baseline_alloc_MiB=base / 2**20,
                       peak_minus_baseline_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20,
                       reserved_MiB=torch.cuda.memory_reserved() / 2**20)
        ref[be] = (float(e), f)
    parity = {}
    for be in BACKENDS:
        dE = abs(ref[be][0] - ref["e3nn"][0]); dF = float(np.abs(ref[be][1] - ref["e3nn"][1]).max())
        parity[be] = dict(E_eV=ref[be][0], abs_dE_eV=dE, abs_dE_per_atom_eV=dE / nat, max_abs_dF_eVA=dF,
                          max_abs_F_ref=float(np.abs(ref["e3nn"][1]).max()),
                          pass_1e12=bool(dE / nat <= PARITY_E_PER_ATOM and dF <= PARITY_F),
                          valid=bool(dE / nat <= PARITY_HARD and dF <= PARITY_HARD))
    print("PARITY", name, json.dumps({be: {k: parity[be][k] for k in ("abs_dE_per_atom_eV", "max_abs_dF_eVA", "valid")} for be in BACKENDS}), flush=True)
    for blk in range(NBLOCKS[name]):
        order = BACKENDS if blk % 2 == 0 else BACKENDS[::-1]
        for be in order:
            for bn, (fn, pre) in Bs[be].items():
                timed(fn, pre)  # 1 untimed re-warm after backend switch
                for _ in range(BLOCK): times[be][bn].append(timed(fn, pre))
    d = dict(n_atoms=nat, n_edges=nedge, memory=mem, first_call_this_system_ms=first, parity=parity,
             stats={be: {bn: stats(v) for bn, v in times[be].items()} for be in BACKENDS}, raw_ms=times)
    d["speedup_median_vs_e3nn"] = {be: {bn: d["stats"]["e3nn"][bn]["median"] / d["stats"][be][bn]["median"] for bn in BNAMES} for be in BACKENDS}
    out["systems"][name] = d
    print("RESULT", name, nat, nedge, json.dumps({be: {bn: round(s["median"], 3) for bn, s in d["stats"][be].items()} for be in BACKENDS}),
          "mem", json.dumps({be: round(m["peak_minus_baseline_MiB"], 1) for be, m in mem.items()}), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
out["nvidia_smi_end"] = nvsmi()
json.dump(out, open(out_path, "w"), indent=1)
print("DONE", out_path, flush=True)
