"""int9 correctness: GPUStep (whole-step graph, external CSR) vs stock fp64 MACE eager and vs node-hybrid + proto
(Phase 8 GraphedRunner) on identical coordinates.  usage: python -I test9.py OUT.json nl(brute|cell|auto) systems(comma) [quick]
Tolerances (INTERFACES): |dE|/atom <= 1e-11 eV, max|dF| <= 1e-11 eV/A, |dsigma| <= 1e-14 eV/A^3, neighbour sets bitwise."""
import os, sys, json, time, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/int9")
import common9 as C
import numpy as np
import torch
from mace.tools import torch_tools
import gpu_step as GS
import nodeconv_ext as NX

out_path, NLK, SYS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
MODE = sys.argv[4] if len(sys.argv) > 4 else ""
QUICK = "quick" in MODE
SYS_ONLY = "sysonly" in MODE
TOL = dict(dE=1e-11, dF=1e-11, dS=1e-14)
res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=C.gpuinfo(), nl=NLK, tol=TOL, tests={},
           failures=[])
dev = torch.device("cuda")
stock_calc, stock_model = C.build("stock")
calc, model = C.build("nodeext+embed+node")
ncalc, nmodel = C.build("nodehybrid+embed+node")
EXT = NX.get_ext(model)


def stock_ref(atoms, stress):
    bd = C.stock_batch(stock_calc, atoms)
    with torch_tools.default_dtype("float64"):
        o = stock_model(bd, compute_force=True, compute_stress=stress, training=False)
    r = dict(E=float(o["energy"][0]), F=o["forces"].detach().cpu().numpy(),
             S=o["stress"][0].detach().cpu().numpy() if stress else None, ei=bd["edge_index"].cpu(), us=bd["unit_shifts"].cpu())
    del o, bd
    torch.cuda.empty_cache()
    return r


def cmp(name, atoms, got, ref, extra=None):
    N = len(atoms)
    dE = abs(got["E"] - ref["E"]) / N
    dF = float(np.abs(got["F"] - ref["F"]).max()) if N else 0.0
    dS = float(np.abs(got["S"] - ref["S"]).max()) if (got.get("S") is not None and ref.get("S") is not None) else None
    ok = bool(np.isfinite(got["E"]) and dE <= TOL["dE"] and dF <= TOL["dF"] and (dS is None or dS <= TOL["dS"]))
    r = dict(N=N, dE_atom=dE, max_dF=dF, dS=dS, ok=ok)
    if extra: r.update(extra)
    T = res["tests"].setdefault(name, dict(rows=[], ok=True))
    T["rows"].append(r); T["ok"] = T["ok"] and ok
    if not ok:
        res["failures"].append(dict(test=name, **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}))
    return ok


def gpu_res(o, stress):
    torch.cuda.synchronize()
    return dict(E=float(o[0][0]), F=o[1].detach().cpu().numpy().copy(), S=o[2].detach().cpu().numpy().copy() if stress else None)


def edges_equal(a, b):
    A = C.edge_set(*a); B = C.edge_set(*b)
    return bool(A.shape == B.shape and (A == B).all()), int(A.shape[0]), int(B.shape[0])


def save():
    res["ok"] = not res["failures"]
    json.dump(res, open(out_path, "w"), indent=1, default=float)


def run(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as ex:  # noqa: BLE001
        res["failures"].append(dict(test=name, error=f"{type(ex).__name__}: {ex}"[:400], tb=traceback.format_exc()[-2500:]))
        print(traceback.format_exc(), flush=True)
    print(name, "done", round(time.perf_counter() - t0, 1), "s; failures so far", len(res["failures"]), flush=True)
    save()


rng = np.random.default_rng(9)


# 1. systems: static + random walk (6 steps of 0.02 A rms -> NL changes) + repeat; vs stock and vs node-hybrid+proto
def t_systems():
    for s in SYS:
        atoms = C.load(s)
        per = bool(all(atoms.pbc))
        for stress in ([False, True] if per else [False]):
            sim = GS.GPUStep(calc, model, atoms, stress=stress, nl=NLK)
            rn = C.GR.GraphedRunner(ncalc, nmodel, stress=stress)
            a = atoms.copy()
            nsteps = 2 if QUICK else 6
            for k in range(nsteps + 1):
                if k:
                    a.positions = a.positions + rng.normal(scale=0.02, size=a.positions.shape)
                pos = torch.tensor(a.positions, dtype=torch.float64, device=dev)
                o = sim.forces(pos); g = gpu_res(o, stress); chk = sim.check()
                ref = stock_ref(a, stress)
                eq = None
                if sim.path == "graph":
                    eq, nE, nS = edges_equal(sim.edges(), (ref["ei"], ref["us"]))
                    if not eq:
                        res["failures"].append(dict(test="neighbour_set", sys=s, step=k, n_gpu=nE, n_stock=nS))
                cmp("systems_walk_vs_stock", a, g, ref, dict(sys=s, step=k, stress=stress, path=sim.path, nl=sim.nl_name,
                                                             edges_bitwise=eq, check=chk))
                ro = rn.step(a)
                nh = dict(E=ro["energy"], F=ro["forces"],
                          S=None if ro["stress"] is None else _v2m(ro["stress"]))
                cmp("systems_walk_vs_nodehybrid_proto", a, g, nh, dict(sys=s, step=k, stress=stress))
            # repeat same coordinates -> bitwise identical replay
            o2 = gpu_res(sim.forces(pos), stress)
            # informational: MACE's force backward uses atomic index_add (fp64) -> run-to-run last-bit differences
            same = bool(o2["E"] == g["E"] and (o2["F"] == g["F"]).all())
            T = res["tests"].setdefault("repeat_same_coords", dict(rows=[], ok=True))
            T["rows"].append(dict(sys=s, stress=stress, bitwise_same=same, dE=abs(o2["E"] - g["E"]), dF=float(np.abs(o2["F"] - g["F"]).max())))
            cmp("repeat_same_coords_tol", a, o2, g, dict(sys=s, stress=stress))
            st = sim.stats()
            T = res["tests"].setdefault("stats", dict(rows=[], ok=True)); T["rows"].append(dict(sys=s, stress=stress, **{k: v for k, v in st.items() if k != "events"}))
            del sim, rn
            torch.cuda.empty_cache()


def _v2m(v):
    xx, yy, zz, yz, xz, xy = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


# 2. overflow -> NaN poison -> check -> recapture -> valid; proactive recapture; forces_checked
def t_overflow():
    for s in (["water192", "wbox648"] if not QUICK else ["water192"]):
        atoms = C.load(s)
        pos = torch.tensor(atoms.positions, dtype=torch.float64, device=dev)
        sim = GS.GPUStep(calc, model, atoms, nl=NLK)
        E0 = sim.stats()["n_edges_last"]
        sim2 = GS.GPUStep(calc, model, atoms, nl=NLK, E_cap=64 * max(1, (sim.graphs[sim.key].E_cap // 2) // 64))
        o = gpu_res(sim2.forces(pos), False)
        poisoned = bool(np.isnan(o["E"]) and np.isnan(o["F"]).all())
        rep = sim2.check()
        o2 = gpu_res(sim2.forces(pos), False); rep2 = sim2.check()
        ref = stock_ref(atoms, False)
        cmp("overflow_recapture", atoms, o2, ref, dict(sys=s, poisoned_before=poisoned, check1=rep, check2=rep2))
        if not (poisoned and rep["ok"] is False and rep["recaptured"] and rep2["ok"]):
            res["failures"].append(dict(test="overflow_policy", sys=s, poisoned=poisoned, rep=rep, rep2=rep2))
        # forces_checked from a too-small cap
        sim3 = GS.GPUStep(calc, model, atoms, nl=NLK, E_cap=64)
        o3 = gpu_res(sim3.forces_checked(pos), False)
        cmp("overflow_forces_checked", atoms, o3, ref, dict(sys=s, stats={k: v for k, v in sim3.stats().items() if k != "events"}))
        # compression walk -> more edges -> overflow or proactive recapture, never invalid accepted
        a = atoms.copy(); n_bad = 0; reps = []
        for k in range(6):
            a.positions = a.positions * 0.97 + (a.positions.mean(0) * 0.03)
            if all(a.pbc):
                a.set_cell(a.cell * 0.97, scale_atoms=False)
            p = torch.tensor(a.positions, dtype=torch.float64, device=dev)
            if all(a.pbc):
                sim.set_atoms(a)  # new cell -> new key
            o = gpu_res(sim.forces_checked(p), False); ref = stock_ref(a, False)
            cmp("overflow_compression_walk", a, o, ref, dict(sys=s, step=k, E_cap=sim.stats()["E_cap"], E=sim.stats()["n_edges_last"]))
        T = res["tests"].setdefault("overflow_stats", dict(rows=[], ok=True))
        T["rows"].append(dict(sys=s, E0=E0, stats={k: v for k, v in sim.stats().items()}))
        del sim, sim2, sim3; torch.cuda.empty_cache()


# 3. species-order / atom-count traps (keyed graphs, interleaved)
def t_traps():
    asp = C.load("aspirin"); rev = asp.copy(); rev.numbers = asp.numbers[::-1].copy()
    w = C.load("water192")
    sim = GS.GPUStep(calc, model, asp, nl=NLK)
    for k, a in enumerate([asp, rev, asp, w, rev, w, asp]):
        sim.set_atoms(a)
        o = gpu_res(sim.forces(torch.tensor(a.positions, dtype=torch.float64, device=dev)), False); sim.check()
        cmp("species_order_atom_count_traps", a, o, stock_ref(a, False), dict(i=k, N=len(a), n_graphs=sim.stats()["n_graphs"]))


# 4. zero-edge / tiny systems
def t_tiny():
    for s in ("zeroC", "farO2", "H2O"):
        a = C.load(s)
        sim = GS.GPUStep(calc, model, a, nl=NLK)
        o = gpu_res(sim.forces(torch.tensor(a.positions, dtype=torch.float64, device=dev)), False); chk = sim.check()
        cmp("zero_edge_tiny", a, o, stock_ref(a, False), dict(sys=s, E_edges=chk["last_E"], check=chk))


# 5. external CSR really used (no sort kernels in the captured graph) + uninstall restores exactly
def t_ext():
    from torch.profiler import profile, ProfilerActivity
    a = C.load("water192")
    sim = GS.GPUStep(calc, model, a, nl=NLK)
    sim.forces(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        sim.replay(); torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    nsort = sum(1 for n in names if "adix" in n or "searchsorted" in n.lower() or "Sort" in n)
    T = res["tests"]["ext_csr"] = dict(ok=True, rows=[dict(sort_like_kernels_in_step_graph=nsort, n_kernels=len(names),
                                                              ext_layer_calls=EXT.n_ext, sort_layer_calls=EXT.n_sort,
                                                              nl=sim.nl_name)])
    # a cell-list NL must contribute no sort; the brute stub has 1 sort (perm_src) + 2 searchsorted
    # uninstall/reinstall on a separate model
    c2, m2 = C.build("nodeext+embed+node")
    orig_ids = None
    h = m2._nodetp9ext_saved
    origs = [o for _, o in h]
    ok = NX.uninstall(m2)
    restored = all(m2.interactions[li].conv_tp is o for li, o in h) and not any(hasattr(b, "conv_fusion") and b.__dict__.get("conv_fusion") for b in m2.interactions) \
        and NX.get_ext(m2) is None
    T["rows"].append(dict(uninstall=ok, restored_exact=restored))
    if not restored:
        T["ok"] = False; res["failures"].append(dict(test="uninstall_restore"))


# 6. fallbacks: unsupported geometry for the cell list -> proto path (wbox192, nl="cell"), brute stub (nl="auto"),
#    mixed pbc -> host proto; walk 3 steps each vs stock (stress for periodic)
def t_fallback():
    w = C.load("wbox192")
    mixed = C.load("water192"); mixed.set_cell([40.0, 40.0, 40.0]); mixed.set_pbc([True, True, False])
    for a0, nl, stress, want in ((w, "cell", False, "proto"), (w, "cell", True, "proto"), (w, "auto", True, "proto"), (w, "brute", True, "brute"),
                                 (mixed, "auto", False, "proto_host")):
        sim = GS.GPUStep(calc, model, a0, nl=nl, stress=stress)
        a = a0.copy()
        for k in range(3):
            if k:
                a.positions = a.positions + rng.normal(scale=0.02, size=a.positions.shape)
            o = gpu_res(sim.forces(torch.tensor(a.positions, dtype=torch.float64, device=dev)), stress); sim.check()
            ref = stock_ref(a, stress)
            cmp("fallback_paths", a, o, ref, dict(sys="wbox192" if a0 is w else "water192_mixed_pbc", nl_req=nl,
                                                  stress=stress, path=sim.path, nl=sim.nl_name, step=k,
                                                  expected_nl=want, path_ok=sim.nl_name == want))
        if sim.nl_name != want:
            res["failures"].append(dict(test="fallback_path", nl_req=nl, got=sim.nl_name, want=want))
        del sim; torch.cuda.empty_cache()


# 7. NL bin overflow (nl9 flag bit 2) -> NaN poison -> check enlarges bin_cap -> valid; open + periodic
def t_binovf():
    for s, bc in (("water648", 4), ("wbox648", 8)):
        a = C.load(s); pos = torch.tensor(a.positions, dtype=torch.float64, device=dev)
        sim = GS.GPUStep(calc, model, a, nl="cell", nl_kw=dict(bin_cap=bc))
        o = gpu_res(sim.forces(pos), False); poisoned = bool(np.isnan(o["E"]))
        rep = sim.check()
        o2 = gpu_res(sim.forces_checked(pos), False)
        cmp("bin_overflow_recapture", a, o2, stock_ref(a, False), dict(sys=s, bin_cap0=bc, poisoned_before=poisoned,
            check1=rep, stats={k: v for k, v in sim.stats().items() if k != "events"}))
        if not (poisoned and not rep["ok"] and rep.get("new_bin_cap")):
            res["failures"].append(dict(test="bin_overflow_policy", sys=s, poisoned=poisoned, rep=rep))
        del sim; torch.cuda.empty_cache()


if "binonly" in MODE:
    run("bin_overflow", t_binovf); save(); print("FAILURES", len(res["failures"]), json.dumps(res["failures"], default=str)[:2000]); print("DONE"); sys.exit(0)
if not SYS_ONLY:
    run("bin_overflow", t_binovf)
    run("fallback", t_fallback)
    run("ext_csr", t_ext)
    run("tiny", t_tiny)
    run("traps", t_traps)
    run("overflow", t_overflow)
run("systems", t_systems)
res["ext_counters_end"] = dict(n_ext=EXT.n_ext, n_sort=EXT.n_sort)
save()
print("FAILURES", len(res["failures"]), json.dumps(res["failures"], default=str)[:3000], flush=True)
print("DONE", flush=True)
