"""prof8 gate (copy of validate7/gate7.py; build via cfg8 incl. nodehybrid; stress tol 1e-14). validate7 integrated correctness gate (copy of scripts/graphed7/test_graphed.py + fusion transforms + eager-fused check).
configs: comma list of <backend>[+embed][+node], e.g. hybrid+embed,hybrid+node,hybrid+embed+node,flash+embed
graphed7 correctness tests: GraphedRunner (e3nn control, hybrid, flash) vs eager stock e3nn fp64 on the stock batch.
usage: python -I test_graphed.py OUT.json [backends,comma]"""
import os, sys, json, time, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/graphed7")
import graphed_runner as GR  # first (pins flashTP graphcap)
sys.path.insert(0, f"{R}/scripts/prof8"); import cfg8
import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.build import molecule
from mace.tools import torch_tools

out_path = sys.argv[1]
BACK = sys.argv[2].split(",")


def transforms_for(cfg):
    parts = cfg.split("+"); tr = []
    if "embed" in parts:
        sys.path.insert(0, f"{R}/scripts/fuse7_embed"); import fused_embed as FE; tr.append(FE.install)
    if "node" in parts:
        sys.path.insert(0, f"{R}/scripts/fuse7_node"); import fused_node as FN; tr.append(FN.install)
    return parts[0], tr, parts[1:]
TOL_E, TOL_F, TOL_S = 1e-11, 1e-11, 1e-14  # phase8 INTERFACES


def gpuinfo():
    import subprocess
    q = "index,name,uuid,clocks.sm,clocks.max.sm"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()


def load(name):
    if name == "H2O":
        return molecule("H2O")
    if name == "single_C":
        return Atoms("C", positions=[[0, 0, 0]])
    if name == "far_pair":
        return Atoms("O2", positions=[[0, 0, 0], [20.0, 0, 0]])
    return ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")


def walk(a0, n=5, sig=0.02, seed=0):
    rng = np.random.default_rng(seed)
    seq = [a0]
    for _ in range(n):
        a = seq[-1].copy(); a.positions = a.positions + rng.normal(0, sig, a.positions.shape); seq.append(a)
    return seq


def compress(a, s):
    b = a.copy()
    if any(b.pbc):
        b.set_cell(np.array(b.cell) * s, scale_atoms=True)
    else:
        c = b.positions.mean(0); b.positions = c + (b.positions - c) * s
    return b


res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=gpuinfo(), torch=torch.__version__,
           flashTP=GR.HB.flashTP_e3nn.__file__, tol=dict(dE_per_atom=TOL_E, max_dF=TOL_F, dstress=TOL_S),
           backends={}, failures=[])
ref_calc, ref_model = GR.build_backend("e3nn")
_refcache = {}


def ref(atoms, tag):
    if tag in _refcache:
        return _refcache[tag]
    bd = GR._stock_build(ref_calc, atoms)
    with torch_tools.default_dtype("float64"):
        o = ref_model(bd, compute_force=True, compute_stress=True, training=False)
    s = o["stress"][0].detach().cpu().numpy()
    r = (float(o["energy"][0]), o["forces"].detach().cpu().numpy().copy(),
         np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]), int(bd["edge_index"].shape[1]))
    _refcache[tag] = r
    return r


def cmp(out, atoms, tag):
    Er, Fr, Sr, Eref = ref(atoms, tag)
    n = max(len(atoms), 1)
    d = np.abs(out["forces"] - Fr)
    row = dict(tag=tag, N=len(atoms), E=out["E"], E_ref_edges=Eref, E_cap=out["E_cap"], recaptured=out["recaptured"],
               dE_per_atom=abs(out["energy"] - Er) / n, max_dF=float(d.max()) if d.size else 0.0,
               rms_dF=float(np.sqrt(((out["forces"] - Fr) ** 2).sum(-1).mean())) if d.size else 0.0,
               maxF=float(np.abs(Fr).max()) if d.size else 0.0,
               finite=bool(np.isfinite(out["energy"]) and np.isfinite(out["forces"]).all()))
    if out["stress"] is not None:
        row["dstress"] = float(np.abs(out["stress"] - Sr).max())
        row["stress_ref_max"] = float(np.abs(Sr).max())
    row["ok"] = bool(row["finite"] and row["dE_per_atom"] <= TOL_E and row["max_dF"] <= TOL_F
                     and row.get("dstress", 0.0) <= TOL_S and out["E"] == Eref)
    return row


SYSTEMS = ["H2O", "aspirin", "water192", "org192", "water648", "org648", "wbox192", "wbox648", "single_C", "far_pair"]
atoms0 = {s: load(s) for s in SYSTEMS}
seqs = {s: walk(atoms0[s]) for s in SYSTEMS}

for bk in BACK:
    B = res["backends"][bk] = {}
    print(f"##### backend {bk}", flush=True)
    kind, tr, fz = transforms_for(bk)
    t0 = time.perf_counter(); calc, model = cfg8.build(bk); B["build_s"] = time.perf_counter() - t0
    B["backend_map"] = calc.graphed7_info["backend_map"]
    B["fusions"] = fz
    B["module_types"] = dict(sh=type(model.spherical_harmonics).__name__, radial=type(model.radial_embedding).__name__,
                             int0_forward=repr(getattr(model.interactions[0], "forward", None))[:120])
    tests = B["tests"] = {}

    def run(name, fn):
        print(f"--- {bk} {name}", flush=True)
        try:
            rows, extra = fn()
            ok = all(r["ok"] for r in rows) and extra.pop("_ok", True)
            tests[name] = dict(ok=ok, rows=rows, **extra)
            worst = dict(dE=max(r["dE_per_atom"] for r in rows), dF=max(r["max_dF"] for r in rows),
                         dS=max([r.get("dstress", 0) for r in rows]))
            tests[name]["worst"] = worst
            print(f"    ok={ok} n={len(rows)} worst={worst} extra={ {k: v for k, v in extra.items() if k != 'events'} }",
                  flush=True)
            if not ok:
                res["failures"].append(f"{bk}/{name}")
        except Exception as ex:
            tests[name] = dict(ok=False, error=f"{type(ex).__name__}: {ex}", traceback=traceback.format_exc())
            res["failures"].append(f"{bk}/{name}: {type(ex).__name__}: {str(ex)[:200]}")
            print(traceback.format_exc(), flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

    # T0: eager (no graph) fused model on the stock batch vs eager stock e3nn
    def t_eager():
        rows = []
        for s_ in SYSTEMS:
            for i in (0, 3):
                a = seqs[s_][i]; bd = GR._stock_build(calc, a)
                with torch_tools.default_dtype("float64"):
                    o = model(bd, compute_force=True, compute_stress=True, training=False)
                st = o["stress"][0].detach().cpu().numpy()
                out = dict(energy=float(o["energy"][0]), forces=o["forces"].detach().cpu().numpy(), recaptured=False,
                           E=int(bd["edge_index"].shape[1]), E_cap=None,
                           stress=np.array([st[0, 0], st[1, 1], st[2, 2], st[1, 2], st[0, 2], st[0, 1]]))
                rows.append(cmp(out, a, f"{s_}/walk{i}"))
        return rows, {}
    run("eager_fused_vs_stock", t_eager)

    # T1: all systems, 5-step 0.02 A walk (coords + neighbour lists change) + 3 repeated calls; one runner (LRU 4)
    def t_systems():
        rn = GR.GraphedRunner(calc, model)
        rows = []
        for s in SYSTEMS:
            for i, a in enumerate(seqs[s]):
                rows.append(cmp(rn.step(a), a, f"{s}/walk{i}"))
            for k in range(3):
                rows.append(cmp(rn.step(seqs[s][0]), seqs[s][0], f"{s}/walk0"))
        st = rn.stats()
        return rows, dict(stats=st, events=rn.events, edges={s: [r["E"] for r in rows if r["tag"].startswith(s + "/")]
                                                              for s in SYSTEMS})

    # T2: stress (periodic boxes) + walk + forced recapture with cell compression and back
    def t_stress():
        rn = GR.GraphedRunner(calc, model, stress=True)
        rows = []
        for s in ["wbox192", "wbox648"]:
            for i, a in enumerate(seqs[s]):
                rows.append(cmp(rn.step(a), a, f"{s}/walk{i}"))
        a0 = atoms0["wbox192"]; ac = compress(a0, 0.93)
        for a, tag in [(ac, "wbox192/c0.93"), (a0, "wbox192/walk0"), (ac, "wbox192/c0.93")]:
            rows.append(cmp(rn.step(a), a, tag))
        rec = [r["recaptured"] for r in rows[-3:]]
        # molecule with stress=True (non-periodic: stress must be 0 like eager)
        for a, tag in [(atoms0["aspirin"], "aspirin/walk0"), (atoms0["single_C"], "single_C/walk0")]:
            rows.append(cmp(rn.step(a), a, tag))
        return rows, dict(stats=rn.stats(), events=rn.events, recapture_flags=rec, _ok=rec == [True, False, False])

    # T3: forced recapture without stress: compress water192 so E > E_cap, back, again
    def t_recapture():
        rn = GR.GraphedRunner(calc, model)
        a0 = atoms0["water192"]; ac = compress(a0, 0.88)
        seq = [(a0, "water192/walk0"), (ac, "water192/c0.88"), (a0, "water192/walk0"), (ac, "water192/c0.88"),
               (seqs["water192"][3], "water192/walk3")]
        rows = [cmp(rn.step(a), a, t) for a, t in seq]
        rec = [r["recaptured"] for r in rows]
        st = rn.stats()
        return rows, dict(stats=st, events=rn.events, recapture_flags=rec,
                          _ok=rec == [False, True, False, False, False] and st["recaptures"] == 1)

    # T4: former bug (a): same N, different species order -> must not reuse the graph
    def t_species():
        rn = GR.GraphedRunner(calc, model)
        a = atoms0["aspirin"]; ar = a.copy(); ar.numbers = a.numbers[::-1].copy()
        ap = a.copy(); ap.numbers = np.random.default_rng(1).permutation(a.numbers)
        seq = [(a, "aspirin/walk0"), (ar, "aspirin_rev"), (a, "aspirin/walk0"), (ap, "aspirin_perm"),
               (ar, "aspirin_rev"), (seqs["aspirin"][2], "aspirin/walk2")]
        rows = [cmp(rn.step(x), x, t) for x, t in seq]
        st = rn.stats()
        return rows, dict(stats=st, events=rn.events, _ok=st["captures"] == 3 and st["recaptures"] == 0)

    # T5: former bug (b): different N then original N; interleaved A,B,A,B with walks
    def t_diffN():
        rn = GR.GraphedRunner(calc, model)
        A, Bs = seqs["aspirin"], seqs["water192"]
        seq = [(A[0], "aspirin/walk0"), (Bs[0], "water192/walk0"), (A[0], "aspirin/walk0")]
        for i in range(1, 5):
            seq += [(A[i], f"aspirin/walk{i}"), (Bs[i], f"water192/walk{i}")]
        seq += [(atoms0["H2O"], "H2O/walk0"), (A[0], "aspirin/walk0")]
        rows = [cmp(rn.step(x), x, t) for x, t in seq]
        st = rn.stats()
        return rows, dict(stats=st, events=rn.events, _ok=st["captures"] == 3 and st["recaptures"] == 0)

    # T6: LRU eviction (max_graphs=2), cycle A,B,C,A,B
    def t_lru():
        rn = GR.GraphedRunner(calc, model, max_graphs=2)
        names = ["aspirin", "H2O", "org192", "aspirin", "H2O"]
        rows = [cmp(rn.step(atoms0[s]), atoms0[s], f"{s}/walk0") for s in names]
        st = rn.stats()
        return rows, dict(stats=st, _ok=st["evictions"] == 3 and st["n_graphs"] == 2)

    # T7: stock-batch builder option (validation path)
    def t_stock():
        rn = GR.GraphedRunner(calc, model, builder="stock", stress=True)
        rows = []
        for s in ["aspirin", "wbox192"]:
            for i in (0, 1, 2):
                rows.append(cmp(rn.step(seqs[s][i]), seqs[s][i], f"{s}/walk{i}"))
        return rows, dict(stats=rn.stats())

    for nm, fn in [("systems_walk_repeat", t_systems), ("stress_periodic_recapture", t_stress),
                   ("forced_recapture", t_recapture), ("bug_a_species_order", t_species),
                   ("bug_b_diffN_interleaved", t_diffN), ("lru_eviction", t_lru), ("stock_builder", t_stock)]:
        run(nm, fn)
        torch.cuda.empty_cache()
    del calc, model
    torch.cuda.empty_cache()

res["gpu_end"] = gpuinfo()
json.dump(res, open(out_path, "w"), indent=1)
print("DONE failures:", res["failures"], flush=True)
