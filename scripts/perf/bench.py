"""MACE-OFF23 medium (fp64) latency benchmark: e3nn vs FlashTP adapter, interleaved blocks in ONE process.
Usage: python -I bench.py <out_json> [systems...]
Backends: e3nn (unmodified), flash (adapter; upstream std::cout per forward goes to job stdout = log file, as shipped),
          flash_null (same modules; fd 1 dup2'd to /dev/null during timed blocks, to isolate the print cost).
Boundaries:
  graph_cpu   : config_from_atoms + AtomicData.from_config (CPU, incl. neighbour list)
  graph_total : calc._atoms_to_batch + clone + dtype cast + to_dict (as in MACECalculator.calculate; includes H2D)
  fwd         : model(batch_dict, compute_force=False)   (positions.requires_grad set by MACE)
  bwd         : torch.autograd.grad(energy, positions) after an untimed fwd (same call MACE's compute_forces makes)
  fwd_bwd     : model(batch_dict, compute_force=True)    (what the calculator calls)
  ase_total   : calc.reset(); atoms.get_potential_energy(); atoms.get_forces()
"""
import os, sys, json, time, contextlib, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flash_proto"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np, torch
from mace.calculators import MACECalculator
from mace import data as mace_data
from mace.tools import torch_tools
import flashtp_adapter as FA
from systems import get_system

R = os.environ["R"]; out_path = sys.argv[1]
names = sys.argv[2:] or ["H2O", "water192", "water648", "water1536"]
WARMUP = int(os.environ.get("WARMUP", 10))
BLOCK = int(os.environ.get("BLOCK", 10))
NBLOCKS = {"H2O": 6, "water192": 6, "water648": 6, "water1536": 4}
sync = torch.cuda.synchronize

calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda",
                      default_dtype="float64", compute_stress=False)
model = calc.models[0]


@contextlib.contextmanager
def stdout_to_null(on):
    if not on:
        yield; return
    sys.stdout.flush(); saved = os.dup(1); nul = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nul, 1)
    try:
        yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(saved); os.close(nul)


def make_batch_dict(atoms):
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v): b[k] = v.to(dtype=torch.float64)
    return b.to_dict()


def graph_cpu(atoms):
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    with torch_tools.default_dtype("float64"):
        cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
        return mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)


def timed(fn, pre=None):
    if pre is not None: pre()
    sync(); t0 = time.perf_counter(); fn(); sync(); return (time.perf_counter() - t0) * 1e3


def boundaries(atoms, bd):
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
    return dict(graph_cpu=(lambda: graph_cpu(atoms), None), graph_total=(lambda: make_batch_dict(atoms), None),
                fwd=(fwd, None), bwd=(bwd, pre_bwd), fwd_bwd=(fwd_bwd, None), ase_total=(ase_total, None))


def stats(x):
    x = np.asarray(x); p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=int(x.size), median=float(p50), p10=float(p10), p90=float(p90), min=float(x.min()),
                iqr=float(p75 - p25), mean=float(x.mean()))


def nvsmi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                           "--format=csv"], capture_output=True, text=True).stdout.strip()


out = dict(slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=torch.cuda.get_device_name(0),
           nvidia_smi_start=nvsmi(), torch=torch.__version__, model_dtype=str(next(model.parameters()).dtype),
           warmup=WARMUP, block=BLOCK, nblocks=NBLOCKS, setup={}, systems={})

# ---- one-time setup costs ----
w648 = get_system("water648"); a0 = w648.copy(); a0.calc = calc
sync(); t0 = time.perf_counter(); calc.reset(); a0.get_potential_energy(); a0.get_forces(); sync()
out["setup"]["e3nn_first_ase_call_water648_ms"] = (time.perf_counter() - t0) * 1e3
sync(); t0 = time.perf_counter(); calc.reset(); a0.get_potential_energy(); a0.get_forces(); sync()
out["setup"]["e3nn_second_ase_call_water648_ms"] = (time.perf_counter() - t0) * 1e3
sync(); t0 = time.perf_counter(); saved = FA.install(model); sync()
out["setup"]["FA_install_first_ms"] = (time.perf_counter() - t0) * 1e3
for k in ("first", "second", "third"):
    sync(); t0 = time.perf_counter(); calc.reset(); a0.get_potential_energy(); a0.get_forces(); sync()
    out["setup"][f"flash_{k}_ase_call_water648_ms"] = (time.perf_counter() - t0) * 1e3
FA.uninstall(model, saved)
inst = []
for _ in range(3):
    sync(); t0 = time.perf_counter(); s = FA.install(model); sync(); inst.append((time.perf_counter() - t0) * 1e3)
    FA.uninstall(model, s)
out["setup"]["FA_install_repeat_ms"] = inst
sync(); t0 = time.perf_counter(); s = FA.install(model); sync(); out["setup"]["FA_install_final_ms"] = (time.perf_counter() - t0) * 1e3
flash_mods = {li: model.interactions[li].conv_tp for li, _ in s}
orig_mods = {li: o for li, o in s}
FA.uninstall(model, s)
print("SETUP", json.dumps(out["setup"]), flush=True)


def set_backend(name):  # pointer swap between pre-built module sets (same effect as FA.install/uninstall)
    for li in orig_mods:
        blk = model.interactions[li]
        if name == "e3nn":
            blk.conv_tp = orig_mods[li]
            if hasattr(blk, "conv_fusion"): del blk.conv_fusion
        else:
            blk.conv_tp = flash_mods[li]; blk.conv_fusion = True
    assert FA.is_installed(model) == ([False, False] if name == "e3nn" else [True, True])


BACKENDS = ["e3nn", "flash", "flash_null"]
for name in names:
    atoms = get_system(name); nat = len(atoms)
    bd = make_batch_dict(atoms); nedge = int(bd["edge_index"].shape[1])
    times = {be: {b: [] for b in ["graph_cpu", "graph_total", "fwd", "bwd", "fwd_bwd", "ase_total"]} for be in BACKENDS}
    ref = {}
    mem = {}
    for be in BACKENDS:  # warmup + correctness spot-check + peak memory
        set_backend(be)
        with stdout_to_null(be == "flash_null"):
            B = boundaries(atoms, bd)
            for _ in range(WARMUP):
                for bn, (fn, pre) in B.items(): timed(fn, pre)
            a = atoms.copy(); a.calc = calc; calc.reset(); sync()
            torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            e = a.get_potential_energy(); f = a.get_forces().copy(); sync()
            mem[be] = dict(peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20, baseline_alloc_MiB=base / 2**20,
                           peak_minus_baseline_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20,
                           reserved_MiB=torch.cuda.memory_reserved() / 2**20)
            ref[be] = (e, f)
    for blk in range(NBLOCKS[name]):
        order = BACKENDS if blk % 2 == 0 else BACKENDS[::-1]
        for be in order:
            set_backend(be)
            with stdout_to_null(be == "flash_null"):
                B = boundaries(atoms, bd)
                for bn, (fn, pre) in B.items():
                    timed(fn, pre)  # 1 untimed re-warm after backend switch
                    for _ in range(BLOCK): times[be][bn].append(timed(fn, pre))
    set_backend("e3nn")
    d = dict(n_atoms=nat, n_edges=nedge, memory=mem,
             dE_flash_vs_e3nn_eV=float(ref["flash"][0] - ref["e3nn"][0]),
             max_dF_flash_vs_e3nn=float(np.abs(ref["flash"][1] - ref["e3nn"][1]).max()),
             stats={be: {bn: stats(v) for bn, v in times[be].items()} for be in BACKENDS}, raw_ms=times)
    d["speedup_median_e3nn_over_flash"] = {bn: d["stats"]["e3nn"][bn]["median"] / d["stats"]["flash"][bn]["median"] for bn in times["e3nn"]}
    d["speedup_median_e3nn_over_flash_null"] = {bn: d["stats"]["e3nn"][bn]["median"] / d["stats"]["flash_null"][bn]["median"] for bn in times["e3nn"]}
    out["systems"][name] = d
    print("RESULT", name, nat, nedge, json.dumps({be: {bn: round(s["median"], 3) for bn, s in d["stats"][be].items()} for be in BACKENDS}),
          "mem", json.dumps(mem), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
out["nvidia_smi_end"] = nvsmi()
json.dump(out, open(out_path, "w"), indent=1)
print("DONE", out_path, flush=True)
