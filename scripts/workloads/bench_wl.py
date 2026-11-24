"""Workload correctness + timing harness (MACE-OFF23 medium, fp64), all backends in ONE process.
Usage: python -I bench_wl.py <out_json> <backends comma-list, e3nn first> <systems...>
Env: MODE=corr|full (default full), WARMUP=10, BLOCK=10, NBLOCKS=7, BACKEND_SET=v1|v2.
Correctness (per system, vs e3nn in same process): fixed config + 5-step random-walk sequence (N(0,0.02 A) per step,
seed fixed per system), same calculator objects reused; 5 repeat calls on fixed config; stress for pbc systems.
Timing boundaries identical to scripts/perf2/bench4.py: graph_cpu (config_from_atoms+AtomicData.from_config),
graph_total (calc._atoms_to_batch+_clone_batch+fp64 cast), fwd (model(compute_force=False)), bwd (autograd.grad of
energy wrt positions; forward excluded via pre-hook), fwd_bwd (model(compute_force=True)), ase_total
(calc.reset(); get_potential_energy(); get_forces()), each bracketed by torch.cuda.synchronize(). Interleaved blocks
of BLOCK samples per (backend, boundary), backend order alternating per block, 1 untimed re-warm per switch."""
import os, sys, json, time, subprocess, traceback
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/workloads")
import backends_wl as BW   # flash imports happen inside builders
from systems2 import load_system
import numpy as np, torch
from mace import data as mace_data
from mace.tools import torch_tools
from mace.data.neighborhood import get_neighborhood

out_path = sys.argv[1]; BACKENDS = sys.argv[2].split(","); names = sys.argv[3:]
assert BACKENDS[0] == "e3nn"
MODE = os.environ.get("MODE", "full")
WARMUP = int(os.environ.get("WARMUP", 10)); BLOCK = int(os.environ.get("BLOCK", 10)); NBLOCKS = int(os.environ.get("NBLOCKS", 7))
GATE = 1e-9            # |dE|/atom (eV) and max|dF| (eV/A) over all configs: above -> backend invalid for timing
sync = torch.cuda.synchronize
BNAMES = ["graph_cpu", "graph_total", "fwd", "bwd", "fwd_bwd", "ase_total"]


def nvsmi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def save():
    json.dump(out, open(out_path, "w"), indent=1)


out = dict(slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=torch.cuda.get_device_name(0),
           nvidia_smi_start=nvsmi(), torch=torch.__version__, backends=BACKENDS, mode=MODE, warmup=WARMUP, block=BLOCK,
           nblocks=NBLOCKS, gate=GATE, backend_set=BW.BACKEND_SET, setup={}, systems={}, argv=sys.argv)
import mace, cuequivariance
out["mace"] = mace.__version__; out["cuequivariance"] = cuequivariance.__version__

calcs = {}
for be in BACKENDS:
    try:
        calcs[be], out["setup"][be] = BW.build(be)
    except Exception as e:
        out["setup"][be] = dict(build_error=traceback.format_exc()[-3000:]); print("BUILD FAIL", be, e, flush=True)
BACKENDS = [b for b in BACKENDS if b in calcs]
tsec = BW.tsec
# setup cost: first/second/third ASE E+F call on the first system (first includes lazy init / JIT / CG build)
a0 = load_system(names[0])
for be in BACKENDS:
    a = a0.copy(); a.calc = calcs[be]
    for k in ("first", "second", "third"):
        _, out["setup"][be][f"{k}_ase_call_{names[0]}_ms"] = tsec(lambda: (calcs[be].reset(), a.get_potential_energy(), a.get_forces()))
    s = out["setup"][be]; s["setup_total_ms"] = s["construct_ms"] + s[f"first_ase_call_{names[0]}_ms"]
print("SETUP", json.dumps({be: {k: round(v, 1) for k, v in s.items() if k.endswith("_ms")} for be, s in out["setup"].items()}), flush=True)


def edgeset(at):
    ei, sh, *_ = get_neighborhood(at.positions, 5.0, pbc=tuple(bool(x) for x in at.pbc),
                                  cell=np.array(at.cell) if at.pbc.any() else None)
    return {(int(i), int(j), *map(int, np.round(s @ np.linalg.inv(np.array(at.cell)) if at.pbc.any() else (0, 0, 0))))
            for i, j, s in zip(ei[0], ei[1], sh)}


def evaluate(be, at, stress):
    c = calcs[be]; a = at.copy(); a.calc = c; c.reset()
    r = dict(E=float(a.get_potential_energy()), F=a.get_forces().copy())
    if stress:
        ip = list(c.implemented_properties)
        c.compute_stress = True; c.implemented_properties = ip + ["stress"]; c.reset()   # calc built with compute_stress=False drops "stress"
        try:
            a2 = at.copy(); a2.calc = c
            r["S"] = a2.get_stress(voigt=False).copy(); r["E_s"] = float(a2.get_potential_energy()); r["F_s"] = a2.get_forces().copy()
        except Exception:
            r["S_err"] = traceback.format_exc()[-2500:]
        finally:
            c.compute_stress = False; c.implemented_properties = ip; c.reset()
    return r


def correctness(name, atoms):
    nat = len(atoms); pbc = bool(atoms.pbc.any())
    seed = 1000 + sum(map(ord, name)); rng = np.random.default_rng(seed)
    confs = [atoms.copy()]
    for _ in range(5):
        b = confs[-1].copy(); b.positions = b.positions + rng.normal(scale=0.02, size=b.positions.shape); confs.append(b)
    es = [edgeset(c) for c in confs]
    seqinfo = dict(seed=seed, noise_A=0.02, n_edges=[len(e) for e in es],
                   edges_added_vs_prev=[None] + [len(es[i] - es[i - 1]) for i in range(1, 6)],
                   edges_removed_vs_prev=[None] + [len(es[i - 1] - es[i]) for i in range(1, 6)])
    res = {be: [evaluate(be, c, pbc) for c in confs] for be in BACKENDS}   # same calc objects through the sequence
    rep = {}
    for be in BACKENDS:
        Es, Fs = [], []
        for _ in range(5):
            r = evaluate(be, atoms, False); Es.append(r["E"]); Fs.append(r["F"])
        rep[be] = dict(E_range_eV=float(max(Es) - min(Es)), max_abs_dF_vs_call0=float(max(np.abs(f - Fs[0]).max() for f in Fs)),
                       max_abs_dF_vs_seq0=float(max(np.abs(f - res[be][0]["F"]).max() for f in Fs)))
    corr = {}
    for be in BACKENDS:
        rows = []
        for k in range(6):
            r, e = res[be][k], res["e3nn"][k]; dF = r["F"] - e["F"]
            row = dict(step=k, dE_per_atom_eV=abs(r["E"] - e["E"]) / nat, max_abs_dF_eVA=float(np.abs(dF).max()),
                       rms_dF_eVA=float(np.sqrt((dF ** 2).mean())), max_abs_F_ref=float(np.abs(e["F"]).max()))
            if pbc:
                if "S" in r and "S" in e:
                    row["max_abs_dstress_eVA3"] = float(np.abs(r["S"] - e["S"]).max()); row["max_abs_stress_ref_eVA3"] = float(np.abs(e["S"]).max())
                    row["stress_call_dE_per_atom_eV"] = abs(r["E_s"] - e["E_s"]) / nat
                    row["stress_call_max_abs_dF_eVA"] = float(np.abs(r["F_s"] - e["F_s"]).max())
                else:
                    row["stress_error"] = r.get("S_err") or e.get("S_err")
            rows.append(row)
        worst = dict(dE_per_atom_eV=max(x["dE_per_atom_eV"] for x in rows), max_abs_dF_eVA=max(x["max_abs_dF_eVA"] for x in rows),
                     rms_dF_eVA=max(x["rms_dF_eVA"] for x in rows))
        if pbc:
            ds = [x.get("max_abs_dstress_eVA3") for x in rows]
            worst["max_abs_dstress_eVA3"] = None if None in ds else max(ds)
            worst["stress_ok"] = None not in ds and max(ds) <= GATE
        worst["valid"] = bool(worst["dE_per_atom_eV"] <= GATE and worst["max_abs_dF_eVA"] <= GATE)
        worst["pass_1e12"] = bool(worst["dE_per_atom_eV"] <= 1e-12 and worst["max_abs_dF_eVA"] <= 1e-12)
        corr[be] = dict(worst=worst, steps=rows, repeat=rep[be], E_fixed_eV=res[be][0]["E"])
    return dict(sequence=seqinfo, backends=corr)


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
    return dict(n=int(x.size), median=float(p50), p10=float(p10), p90=float(p90), min=float(x.min()), iqr=float(p75 - p25), mean=float(x.mean()))


for name in names:
    atoms = load_system(name); nat = len(atoms)
    bds = {be: make_batch_dict(calcs[be], atoms) for be in BACKENDS}
    d = dict(formula=atoms.get_chemical_formula(), n_atoms=nat, n_edges=int(bds["e3nn"]["edge_index"].shape[1]), pbc=bool(atoms.pbc.any()))
    d["correctness"] = correctness(name, atoms)
    valid = [be for be in BACKENDS if d["correctness"]["backends"][be]["worst"]["valid"]]
    print("CORR", name, nat, d["n_edges"], json.dumps({be: {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in c["worst"].items()}
                                                     for be, c in d["correctness"]["backends"].items()}),
          "repeat", json.dumps({be: c["repeat"] for be, c in d["correctness"]["backends"].items()}), flush=True)
    out["systems"][name] = d; save()
    if MODE != "full":
        continue
    d["timed_backends"] = valid
    Bs = {be: boundaries(calcs[be], atoms, bds[be]) for be in valid}
    times = {be: {b: [] for b in BNAMES} for be in valid}; mem, first = {}, {}
    for be in valid:
        a = atoms.copy(); a.calc = calcs[be]
        _, first[be] = tsec(lambda: (calcs[be].reset(), a.get_potential_energy(), a.get_forces()))
        for _ in range(WARMUP):
            for bn, (fn, pre) in Bs[be].items(): timed(fn, pre)
        calcs[be].reset(); sync(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        a.get_potential_energy(); a.get_forces(); sync()
        mem[be] = dict(peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20, baseline_alloc_MiB=base / 2**20,
                       peak_minus_baseline_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20, reserved_MiB=torch.cuda.memory_reserved() / 2**20)
    for blk in range(NBLOCKS):
        order = valid if blk % 2 == 0 else valid[::-1]
        for be in order:
            for bn, (fn, pre) in Bs[be].items():
                timed(fn, pre)
                for _ in range(BLOCK): times[be][bn].append(timed(fn, pre))
    d.update(memory=mem, first_call_this_system_ms=first, stats={be: {bn: stats(v) for bn, v in times[be].items()} for be in valid}, raw_ms=times)
    d["speedup_median_vs_e3nn"] = {be: {bn: d["stats"]["e3nn"][bn]["median"] / d["stats"][be][bn]["median"] for bn in BNAMES} for be in valid}
    print("RESULT", name, nat, d["n_edges"], json.dumps({be: {bn: round(s["median"], 3) for bn, s in d["stats"][be].items()} for be in valid}),
          "mem", json.dumps({be: round(m["peak_minus_baseline_MiB"], 1) for be, m in mem.items()}), flush=True)
    save()
out["nvidia_smi_end"] = nvsmi(); save()
print("DONE", out_path, flush=True)
