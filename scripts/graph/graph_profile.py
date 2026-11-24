"""Graph-build decomposition for MACECalculator (mace-torch 0.3.17), stock e3nn MACE-OFF23 medium, fp64.
Usage: python -I graph_profile.py <out_json> [systems...]

graph_total = calc._clone_batch(calc._atoms_to_batch(atoms)) + float cast loop + to_dict
            (identical to scripts/perf2/bench4.py make_batch_dict == the pre-model part of calculate()).
Instrumented run replicates _atoms_to_batch / calculate() line-by-line with timers; inner parts of
AtomicData.from_config are timed by wrapping module-level callees (get_neighborhood, matscipy
neighbour_list, atomic_numbers_to_indices, to_one_hot, AtomicData.__init__). Instrumented and plain
runs are interleaved rep by rep.
Then: H2D variants, ase NL alternative, prototype GPU builder (scripts/graph/proto_graph.py) timing + parity.
"""
import os, sys, json, time, subprocess
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_proto"); sys.path.insert(0, f"{R}/scripts/graph")
import numpy as np, torch
from systems import get_system, water_cluster
from mace.calculators import MACECalculator
from mace import data as mace_data
from mace.data import atomic_data as AD, neighborhood as NB
from mace.tools import torch_tools, torch_geometric
import proto_graph

out_path = sys.argv[1]
names = sys.argv[2:] or ["H2O", "water192", "water648", "water1536", "waterbox648_pbc"]
WARMUP = int(os.environ.get("WARMUP", 10)); REPS = int(os.environ.get("REPS", 60))
sync = torch.cuda.synchronize
pc = time.perf_counter


def nvsmi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def water_box(n_side=6, density=1.0, seed=1):
    """Deterministic periodic water box, n_side^3 molecules at `density` g/cm^3 (cubic, pbc=True)."""
    nmol = n_side ** 3
    vol = nmol * 18.01528 * 1.66053906660e-24 / density * 1e24   # A^3
    L = vol ** (1 / 3); sp = L / n_side
    a = water_cluster(n_side, spacing=sp, seed=seed)
    a.positions += sp / 2
    a.set_cell([L, L, L]); a.set_pbc(True)
    return a


def system(name):
    if name == "waterbox648_pbc": return water_box(6)
    return get_system(name)


def stats(x):
    x = np.asarray(x); p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return dict(n=int(x.size), median=float(p50), p10=float(p10), p90=float(p90), mean=float(x.mean()))


# ---------------- wrappers for inner timing -----------------
ACC = {}
def _wrap(mod, name, label):
    f = getattr(mod, name)
    def w(*a, **k):
        t0 = pc(); r = f(*a, **k); ACC[label] = ACC.get(label, 0.0) + (pc() - t0); return r
    setattr(mod, name, w)
    return f
_orig = {}
_orig["gn"] = _wrap(AD, "get_neighborhood", "get_neighborhood")
_orig["nl"] = _wrap(NB, "neighbour_list", "matscipy_neighbour_list")
_orig["idx"] = _wrap(AD, "atomic_numbers_to_indices", "atomic_numbers_to_indices")
_orig["oh"] = _wrap(AD, "to_one_hot", "to_one_hot")
_init = AD.AtomicData.__init__
def _init_w(self, *a, **k):
    t0 = pc(); _init(self, *a, **k); ACC["AtomicData_init"] = ACC.get("AtomicData_init", 0.0) + (pc() - t0)
AD.AtomicData.__init__ = _init_w
INSTR = {"on": True}   # wrappers always on (both runs) -> identical code path; overhead ~1us/call


def make_batch_dict(calc, atoms):          # == bench4 graph_total
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v): b[k] = v.to(dtype=torch.float64)
    return b.to_dict()


def instrumented(calc, atoms):
    """Line-by-line replica of MACECalculator._atoms_to_batch + calculate() pre-model part."""
    T = {}; ACC.clear()
    sync(); t_start = pc(); t0 = t_start
    calc.arrays_keys.update({"charges": calc.charges_key})
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    ctx = torch_tools.default_dtype(calc.default_dtype); ctx.__enter__()
    t1 = pc(); T["py_keyspec_dtypectx"] = t1 - t0; t0 = t1
    config = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
    t1 = pc(); T["config_from_atoms"] = t1 - t0; t0 = t1
    real_graph = mace_data.AtomicData.from_config(config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
    t1 = pc(); fc = t1 - t0; t0 = t1
    ctx.__exit__(None, None, None)
    real_num_atoms = int(real_graph["node_attrs"].shape[0]); real_num_edges = int(real_graph["edge_index"].shape[1])
    if calc.use_compile and not calc._padding_initialized: raise RuntimeError
    target_num_atoms = max(real_num_atoms, calc.pad_num_atoms); target_num_edges = max(real_num_edges, calc.pad_num_edges)
    assert target_num_atoms == real_num_atoms and target_num_edges == real_num_edges
    data_list = [real_graph]
    t1 = pc(); T["py_pad_checks"] = t1 - t0; t0 = t1
    cpu_batch = torch_geometric.Batch.from_data_list(data_list)
    t1 = pc(); T["collate_from_data_list"] = t1 - t0; t0 = t1
    batch_base = cpu_batch.to(calc.device); sync()
    t1 = pc(); T["h2d_to_device"] = t1 - t0; t0 = t1
    _ = len(atoms); _ = calc.pad_num_atoms > 0 or calc.pad_num_edges > 0
    batch = calc._clone_batch(batch_base); sync()
    t1 = pc(); T["clone_batch_gpu"] = t1 - t0; t0 = t1
    model_dtype = next(calc.models[0].parameters()).dtype
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=model_dtype)
    t1 = pc(); T["dtype_cast_loop"] = t1 - t0; t0 = t1
    bd = batch.to_dict(); sync()
    t1 = pc(); T["to_dict"] = t1 - t0
    total = t1 - t_start
    gn = ACC["get_neighborhood"]; nl = ACC["matscipy_neighbour_list"]
    T["nl_matscipy"] = nl
    T["nl_mace_wrapper(cell/dirs/selfedge/shifts)"] = gn - nl
    T["fromcfg_one_hot(indices+to_one_hot)"] = ACC["atomic_numbers_to_indices"] + ACC["to_one_hot"]
    T["fromcfg_AtomicData_init"] = ACC["AtomicData_init"]
    T["fromcfg_other(torch.tensor x~30, det/inv)"] = fc - gn - T["fromcfg_one_hot(indices+to_one_hot)"] - T["fromcfg_AtomicData_init"]
    T = {k: v * 1e3 for k, v in T.items()}
    return T, total * 1e3, bd, cpu_batch


ORDER = ["py_keyspec_dtypectx", "config_from_atoms", "nl_mace_wrapper(cell/dirs/selfedge/shifts)", "nl_matscipy",
         "fromcfg_one_hot(indices+to_one_hot)", "fromcfg_other(torch.tensor x~30, det/inv)", "fromcfg_AtomicData_init",
         "py_pad_checks", "collate_from_data_list", "h2d_to_device", "clone_batch_gpu", "dtype_cast_loop", "to_dict"]


def timeit(fn, reps=REPS, warm=WARMUP):
    for _ in range(warm): fn()
    xs = []
    for _ in range(reps):
        sync(); t0 = pc(); fn(); sync(); xs.append((pc() - t0) * 1e3)
    return stats(xs)


def sorted_edges(bd):
    ei = bd["edge_index"].cpu().numpy(); us = bd["unit_shifts"].cpu().numpy(); sh = bd["shifts"].cpu().numpy()
    order = np.lexsort((us[:, 2], us[:, 1], us[:, 0], ei[1], ei[0]))
    return ei[:, order], us[order], sh[order]


def model_EF(calc, bd):
    d = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in bd.items()}
    with torch_tools.default_dtype("float64"):
        out = calc.models[0](d, compute_force=True, training=False)
    return float(out["energy"].detach().sum()), out["forces"].detach().cpu().numpy()


out = dict(slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=torch.cuda.get_device_name(0),
           nvidia_smi_start=nvsmi(), torch=torch.__version__, warmup=WARMUP, reps=REPS, systems={},
           cpu_affinity=len(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads())
import mace, matscipy, ase
out.update(mace=mace.__version__, matscipy=matscipy.__version__, ase=ase.__version__)
calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64", compute_stress=False)
out["r_max"] = float(calc.r_max)
PB = proto_graph.ProtoBuilder(calc)

for name in names:
    atoms = system(name); res = dict(n_atoms=len(atoms), pbc=[bool(x) for x in atoms.pbc], cell=np.array(atoms.cell).tolist())
    print("==", name, len(atoms), flush=True)
    # warmup both
    for _ in range(WARMUP): make_batch_dict(calc, atoms); instrumented(calc, atoms)
    parts = {k: [] for k in ORDER}; inst_tot = []; plain = []
    for r in range(REPS):
        sync(); t0 = pc(); make_batch_dict(calc, atoms); sync(); plain.append((pc() - t0) * 1e3)
        T, tot, bd, cpu_batch = instrumented(calc, atoms)
        for k in ORDER: parts[k].append(T[k])
        inst_tot.append(tot)
    res["n_edges"] = int(bd["edge_index"].shape[1])
    res["parts"] = {k: stats(v) for k, v in parts.items()}
    res["sum_of_part_medians"] = float(sum(res["parts"][k]["median"] for k in ORDER))
    res["instrumented_total"] = stats(inst_tot)
    res["graph_total_plain"] = stats(plain)
    tens = [v.cpu() for v in cpu_batch.to_dict().values() if torch.is_tensor(v)]
    res["n_tensors"] = len(tens); res["h2d_bytes"] = int(sum(v.numel() * v.element_size() for v in tens))

    # ---- H2D / NL variants (standalone, informational) ----
    v = {}
    # NOTE Data.to()/pin_memory() mutate the Batch in place -> use a shallow copy (copy.copy, ~us) of a CPU template
    import copy
    with torch_tools.default_dtype(calc.default_dtype):
        cfg0 = mace_data.config_from_atoms(atoms, key_specification=mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys), head_name=calc.head)
        g0 = mace_data.AtomicData.from_config(cfg0, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
    cpu_t = torch_geometric.Batch.from_data_list([g0])
    v["h2d_pageable_.to"] = timeit(lambda: copy.copy(cpu_t).to(calc.device))
    v["h2d_pin_memory+to_nonblocking"] = timeit(lambda: copy.copy(cpu_t).pin_memory().to(calc.device, non_blocking=True))
    pin_t = copy.copy(cpu_t).pin_memory()
    v["h2d_prepinned_to_nonblocking(no pin cost)"] = timeit(lambda: copy.copy(pin_t).to(calc.device, non_blocking=True))
    flat = torch.cat([t.reshape(-1).to(torch.float64) for t in tens]).pin_memory()
    v["h2d_single_packed_pinned_buffer"] = timeit(lambda: flat.to(calc.device, non_blocking=True))
    v["collate_from_data_list_alone"] = timeit(lambda: torch_geometric.Batch.from_data_list([g0]))
    gb = copy.copy(cpu_t).to(calc.device)
    v["clone_batch_gpu_alone"] = timeit(lambda: calc._clone_batch(gb))
    cfg = mace_data.config_from_atoms(atoms, key_specification=mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys))
    v["get_neighborhood_alone"] = timeit(lambda: _orig["gn"](cfg.positions, calc.r_max, pbc=cfg.pbc, cell=cfg.cell))
    from ase.neighborlist import primitive_neighbor_list
    def ase_nl():
        ret_cell, ext = proto_graph.mace_cells(cfg.positions, cfg.pbc, cfg.cell, calc.r_max)
        return primitive_neighbor_list("ijS", cfg.pbc, ext, cfg.positions - cfg.positions.min(0) + calc.r_max if not any(cfg.pbc) else cfg.positions, calc.r_max)
    v["ase_primitive_neighbor_list"] = timeit(ase_nl, reps=min(REPS, 20), warm=2)
    res["variants"] = v

    # ---- prototype ----
    pt = timeit(lambda: PB.build(atoms))
    res["proto_build"] = pt
    pos_g = torch.tensor(atoms.positions, device="cuda")
    ret_cell, ext = proto_graph.mace_cells(atoms.positions, tuple(bool(x) for x in atoms.pbc), np.array(atoms.cell), calc.r_max)
    if not any(atoms.pbc):
        res["proto_gpu_nl_only"] = timeit(lambda: PB.neighbour_list(pos_g, (False,) * 3, ext, torch.tensor(ext, device="cuda"), torch.tensor(ret_cell, device="cuda")))
    # parity: batch dict
    ref = make_batch_dict(calc, atoms); new = PB.build(atoms)
    par = dict(same_keys=list(ref.keys()) == list(new.keys()))
    mism = []
    for k in ref:
        a, b = ref[k], new[k]
        if k in ("edge_index", "shifts", "unit_shifts"): continue
        if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device or not torch.equal(a, b): mism.append(k)
    par["nonedge_keys_bitwise_mismatch"] = mism
    e1, u1, s1 = sorted_edges(ref); e2, u2, s2 = sorted_edges(new)
    par["n_edges_ref"] = int(e1.shape[1]); par["n_edges_proto"] = int(e2.shape[1])
    par["edges_bitwise_equal_sorted"] = bool(e1.shape == e2.shape and (e1 == e2).all() and (u1 == u2).all() and (s1 == s2).all())
    par["edge_dtypes"] = [str(ref[k].dtype) + "/" + str(new[k].dtype) for k in ("edge_index", "shifts", "unit_shifts")]
    par["proto_edge_order_equals_stock"] = bool(torch.equal(ref["edge_index"], new["edge_index"]))
    if any(atoms.pbc):
        par["n_nonzero_unit_shift_edges"] = int((np.abs(u1).sum(1) > 0).sum())
        par["max_abs_unit_shift"] = float(np.abs(u1).max())
    E1, F1 = model_EF(calc, ref); E2, F2 = model_EF(calc, new)
    # also proto dict with edges permuted into stock order -> must be bitwise
    par.update(E_ref=E1, E_proto=E2, abs_dE=abs(E1 - E2), abs_dE_per_atom=abs(E1 - E2) / len(atoms),
               max_abs_dF=float(np.abs(F1 - F2).max()), max_abs_F=float(np.abs(F1).max()))
    # second build (cached path) must give the same result
    E3, F3 = model_EF(calc, PB.build(atoms)); par["proto_repeat_abs_dE"] = abs(E3 - E2); par["proto_repeat_max_abs_dF"] = float(np.abs(F3 - F2).max())
    E4, F4 = model_EF(calc, make_batch_dict(calc, atoms))   # stock vs stock: GPU scatter-add (atomic) run-to-run noise floor
    par["stock_repeat_abs_dE"] = abs(E4 - E1); par["stock_repeat_max_abs_dF"] = float(np.abs(F4 - F1).max())
    # moved geometry: perturb positions, rebuild (cache hit), compare to stock
    a2 = atoms.copy(); a2.positions += np.random.default_rng(7).normal(scale=0.05, size=a2.positions.shape)
    r2 = make_batch_dict(calc, a2); n2 = PB.build(a2)
    e1, u1, s1 = sorted_edges(r2); e2, u2, s2 = sorted_edges(n2)
    par["perturbed_edges_bitwise_equal_sorted"] = bool(e1.shape == e2.shape and (e1 == e2).all() and (u1 == u2).all() and (s1 == s2).all())
    par["perturbed_nonedge_mismatch"] = [k for k in r2 if k not in ("edge_index", "shifts", "unit_shifts") and not torch.equal(r2[k], n2[k])]
    Ea, Fa = model_EF(calc, r2); Eb, Fb = model_EF(calc, n2)
    par.update(perturbed_abs_dE=abs(Ea - Eb), perturbed_max_abs_dF=float(np.abs(Fa - Fb).max()))
    res["parity"] = par
    out["systems"][name] = res
    p = res["parts"]
    print(json.dumps({k: round(p[k]["median"], 3) for k in ORDER}), flush=True)
    print("sum", round(res["sum_of_part_medians"], 3), "instr", round(res["instrumented_total"]["median"], 3),
          "plain", round(res["graph_total_plain"]["median"], 3), "proto", round(pt["median"], 3), flush=True)
    print("variants", json.dumps({k: round(x["median"], 3) for k, x in v.items()}), flush=True)
    print("parity", json.dumps(par), flush=True)

out["nvidia_smi_end"] = nvsmi()
json.dump(out, open(out_path, "w"), indent=1)
print("WROTE", out_path)
