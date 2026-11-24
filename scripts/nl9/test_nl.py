"""nl9 correctness tests. usage: python -I test_nl.py OUT.json [quick]
Edge sets (sorted, incl. unit_shifts/shifts) bitwise vs proto_graph and stock get_neighborhood; CSR invariants;
overflow flags; CUDA-graph capture/replay; whole-model (stock e3nn fp64 eager) vs proto / stock batches."""
import os, sys, json, time, traceback, subprocess
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/nl9"); sys.path.insert(0, f"{R}/scripts/nodetp8_bwd")
import numpy as np
import torch
import nlsys
from nlsys import load, canon, stock_nl
import cell_nl
from cell_nl import CellNeighborList, BatchTemplate, round_cap
import proto_graph
from csr import build_csr
from mace.calculators import MACECalculator
from mace.tools import torch_tools

OUT = sys.argv[1]
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
TOL_E, TOL_F, TOL_S = 1e-11, 1e-11, 1e-14
dev = torch.device("cuda")


def gpuinfo():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu=gpuinfo(), torch=torch.__version__,
           tests=[], failures=[])
calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64")
model = calc.models[0]
r_max = float(calc.r_max)
pb = proto_graph.ProtoBuilder(calc)
res["r_max"] = r_max


def rec(name, ok, **kw):
    row = dict(test=name, ok=bool(ok), **kw)
    res["tests"].append(row)
    if not ok:
        res["failures"].append(row)
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps({k: v for k, v in kw.items() if not isinstance(v, list)}),
          flush=True)
    return ok


def make_nl(atoms, E_cap=None, **kw):
    pbc = tuple(bool(x) for x in atoms.pbc)
    if E_cap is None:
        ei, *_ = stock_nl(atoms, r_max)
        E_cap = round_cap(1.2 * ei.shape[1] + 1, 64)
    return CellNeighborList(r_max, pbc, np.array(atoms.cell), len(atoms), E_cap, "cuda", **kw)


def nl_edges(out):
    E = int(out["n_edges"].item())
    Ec = out["edge_index"].shape[1]
    Eu = min(E, Ec)
    return (E, out["edge_index"][:, :Eu].cpu().numpy(), out["unit_shifts"][:Eu].cpu().numpy(),
            out["shifts"][:Eu].cpu().numpy())


def csr_checks(out, N):
    """returns dict of invariant booleans."""
    E = int(out["n_edges"].item()); Ec = out["edge_index"].shape[1]
    ei = out["edge_index"].cpu().numpy(); us = out["unit_shifts"].cpu().numpy(); sh = out["shifts"].cpu().numpy()
    rp = out["rowptr_dst"].cpu().numpy().astype(np.int64); rps = out["rowptr_src"].cpu().numpy()
    rpp = out["rowptr_dst_pad"].cpu().numpy()
    ps = out["perm_src"].cpu().numpy()
    c = {}
    c["rowptr_monotone"] = bool(rp[0] == 0 and np.all(np.diff(rp) >= 0) and rp[N] == E)
    c["rowptr_src_eq_dst"] = bool(np.array_equal(rp, rps))
    dst = ei[1, :E]; src = ei[0, :E]
    c["dst_sorted"] = bool(np.all(np.diff(dst) >= 0))
    c["dst_matches_rowptr"] = bool(np.array_equal(np.repeat(np.arange(N), np.diff(rp)), dst))
    ok = True
    for n in range(N):
        s = src[rp[n]:rp[n + 1]]
        if s.size > 1 and not np.all(np.diff(s) > 0):
            ok = False; break
    c["row_src_strictly_increasing"] = ok
    c["perm_src_is_perm"] = bool(np.array_equal(np.sort(ps), np.arange(Ec)))
    c["perm_src_src_sorted"] = bool(np.all(np.diff(ei[0, ps[:E]]) >= 0)) if E else True
    c["perm_src_row_bounds"] = bool(np.array_equal(np.repeat(np.arange(N), np.diff(rp)), ei[0, ps[:E]]))
    rev = ps[:E]
    c["reverse_edge_map"] = bool(np.array_equal(ei[0, rev], ei[1, :E]) and np.array_equal(ei[1, rev], ei[0, :E])
                                 and np.array_equal(us[rev], -us[:E] + 0.0))
    c["padding"] = bool(np.all(ei[:, E:] == N - 1) and np.array_equal(ps[E:], np.arange(E, Ec))
                        and np.all(np.linalg.norm(sh[E:], axis=1) >= r_max) if Ec > E else True)
    # bitwise equal to nodetp8_bwd.csr.build_csr on the padded edge_index
    ref = build_csr(out["edge_index"], N)
    c["eq_build_csr"] = bool(torch.equal(ref["perm_dst"], out["perm_dst"]) and torch.equal(ref["perm_src"], out["perm_src"])
                             and torch.equal(ref["rowptr_dst"], out["rowptr_dst_pad"])
                             and torch.equal(ref["rowptr_src"], out["rowptr_src_pad"]))
    return c


def edge_compare(atoms, out, tag, with_stock=True, extra=None, stock_tol=None):
    """stock_tol: boundary tests only -- allowed |d - r_max| (A) for edges where matscipy's (opaque) fp64 rounding
    decides differently from proto_graph/torch; everything else must be bitwise equal to stock too."""
    E, ei, us, sh = nl_edges(out)
    ovf = int(out["overflow"].item())
    a = canon(ei, us, sh)
    b = pb.build(atoms)
    p = canon(b["edge_index"].cpu().numpy(), b["unit_shifts"].cpu().numpy(), b["shifts"].cpu().numpy())
    row = dict(N=len(atoms), E=E, E_cap=out["edge_index"].shape[1], overflow=ovf, max_bin=int(out["max_bin"].item()))
    row["eq_proto"] = bool(E == p[0].shape[1] and all(np.array_equal(x, y) for x, y in zip(a, p)))
    row["eq_proto_set"] = bool(E == p[0].shape[1] and np.array_equal(a[0], p[0]) and np.array_equal(a[1], p[1]))
    row["cell_eq_proto"] = bool(torch.equal(out["cell"], b["cell"]) and torch.equal(out["rcell"], b["rcell"])
                                and torch.equal(out["volume"], b["volume"]))
    if with_stock:
        sei, sus, ssh, _ = stock_nl(atoms, r_max)
        s = canon(sei, sus, ssh)
        row["eq_stock_set"] = bool(E == s[0].shape[1] and np.array_equal(a[0], s[0]) and np.array_equal(a[1], s[1]))
        row["eq_stock_shifts"] = bool(row["eq_stock_set"] and np.array_equal(a[2], s[2]))
        if row["eq_stock_set"] and not row["eq_stock_shifts"]:
            row["stock_shift_maxdiff"] = float(np.abs(a[2] - s[2]).max())
        if not row["eq_stock_set"]:   # diagnose: |d - r_max| of the edges in the symmetric difference
            A = set(map(tuple, np.concatenate([a[0].T, a[1]], 1).tolist()))
            B = set(map(tuple, np.concatenate([s[0].T, s[1]], 1).tolist()))
            diff = list(A ^ B)
            C = np.array(atoms.cell) if any(atoms.pbc) else np.zeros((3, 3))
            dd = [abs(np.linalg.norm(atoms.positions[int(t[1])] - atoms.positions[int(t[0])] + np.array(t[2:]) @ C) - r_max)
                  for t in diff]
            row["stock_symdiff"] = len(diff); row["stock_only"] = len(B - A)
            row["stock_symdiff_max_abs_d_minus_rmax"] = float(max(dd))
    stock_ok = (not with_stock) or row["eq_stock_set"] or (
        stock_tol is not None and row["stock_symdiff_max_abs_d_minus_rmax"] <= stock_tol)
    c = csr_checks(out, len(atoms))
    row.update(c)
    if extra:
        row.update(extra)
    ok = ovf == 0 and row["eq_proto"] and row["cell_eq_proto"] and all(c.values()) and \
        stock_ok
    return rec(tag, ok, **row)


def walk(a0, n, sig, seed):
    rng = np.random.default_rng(seed)
    seq = [a0]
    for _ in range(n):
        a = seq[-1].copy(); a.positions = a.positions + rng.normal(0, sig, a.positions.shape); seq.append(a)
    return seq


# ---------------- 0. sum-order check of the distance formula vs torch (d*d).sum(-1) ----------------
try:
    g = torch.Generator(device="cuda").manual_seed(0)
    d = (torch.rand((4_000_000, 3), generator=g, device=dev, dtype=torch.float64) - 0.5) * 10
    ref = (d * d).sum(-1)
    mine = (d[:, 0] * d[:, 0] + d[:, 2] * d[:, 2]) + d[:, 1] * d[:, 1]
    d4 = (d.view(1, 1, -1, 3) * d.view(1, 1, -1, 3)).sum(-1).view(-1)
    rec("d2_sum_order_matches_torch", torch.equal(ref, mine) and torch.equal(d4, mine),
        n=4_000_000, mismatches=int((ref != mine).sum().item()))
except Exception as e:
    rec("d2_sum_order_matches_torch", False, err=repr(e))

# ---------------- 1. edge sets for all systems (+ walks) ----------------
SYS = ["H2O", "aspirin", "org192", "org648", "water192", "water648", "wbox648", "wbox648_outside", "wbox648_tric",
       "wbox648_tric_outside", "wclus2187", "wclus5184", "wbox2187", "wbox2187_tric", "wbox5184",
       "single_C", "far_pair"]
if QUICK:
    SYS = ["H2O", "aspirin", "water192", "wbox648", "wbox648_tric", "single_C"]
NL_CACHE = {}
for name in SYS:
    try:
        a0 = load(name)
        nl = make_nl(a0)
        NL_CACHE[name] = nl
        seq = walk(a0, 0 if QUICK else 3, 0.05, 1)
        for k, a in enumerate(seq):
            out = nl.build(torch.tensor(a.positions, dtype=torch.float64, device=dev))
            torch.cuda.synchronize()
            edge_compare(a, out, f"edges/{name}/walk{k}", extra=dict(nbins=nl.nbins, nb=list(nl.nb), bin_cap=nl.bin_cap))
        # determinism: second build bitwise identical
        o1 = {k: v.clone() for k, v in nl.build(torch.tensor(a0.positions, device=dev)).items()}
        o2 = nl.build(torch.tensor(a0.positions, device=dev))
        rec(f"determinism/{name}", all(torch.equal(o1[k], o2[k]) for k in o1))
    except Exception as e:
        traceback.print_exc(); rec(f"edges/{name}", False, err=repr(e))

# wbox192 (L=12.4 < 3*r_max) and mixed pbc -> NotImplementedError
for name, mod in (("wbox192", None), ("mixed_pbc", "mixed")):
    try:
        a = load("wbox192" if mod is None else "wbox648")
        pbc = tuple(bool(x) for x in a.pbc) if mod is None else (True, True, False)
        try:
            CellNeighborList(r_max, pbc, np.array(a.cell), len(a), 1024, "cuda"); raised = False
        except NotImplementedError as e:
            raised = True; msg = str(e)
        rec(f"unsupported_raises/{name}", raised, msg=msg if raised else "")
    except Exception as e:
        rec(f"unsupported_raises/{name}", False, err=repr(e))

# ---------------- 2. boundary pairs at r_max +- eps ----------------
for periodic in (False, True):
    try:
        a = nlsys.boundary_pairs(r_max, periodic)
        nl = make_nl(a, E_cap=256)
        out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
        edge_compare(a, out, f"boundary/{'pbc' if periodic else 'open'}", extra=dict(npairs=len(a) // 2), stock_tol=1e-13)
        for s in range(3):   # random perturbation of the boundary set (pairs move across r_max)
            b = a.copy(); rng = np.random.default_rng(10 + s)
            b.positions = b.positions + rng.choice([-1, 0, 1], size=b.positions.shape) * 1e-13
            out = nl.build(torch.tensor(b.positions, device=dev)); torch.cuda.synchronize()
            edge_compare(b, out, f"boundary/{'pbc' if periodic else 'open'}/perturb{s}", stock_tol=1e-13)
    except Exception as e:
        traceback.print_exc(); rec(f"boundary/{periodic}", False, err=repr(e))

# dense random boundary stress: many pairs with distances in r_max*(1 +- 1e-14..1e-12) inside a water box
try:
    a0 = load("wbox648"); rng = np.random.default_rng(3)
    a = a0.copy(); p = a.positions.copy()
    for k in range(0, 600, 2):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        p[k + 1] = p[k] + (r_max + rng.choice([-1, 1]) * 10 ** rng.uniform(-14, -12)) * u
    a.positions = p
    nl = make_nl(a)
    out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
    edge_compare(a, out, "boundary/wbox648_300pairs", with_stock=True, stock_tol=1e-13)
    for base_name, lo, hi, tagp in (("wbox648", -16, -14, "ulp"), ("wbox648", -15.7, -15, "ulp2"),
                                    ("water648", -16, -14, "ulp"), ("water648", -15.7, -15, "ulp2")):
        a0 = load(base_name); nl = make_nl(a0, E_cap=round_cap(1.3 * 40000, 64))
        for sd in range(3):
            rng = np.random.default_rng(100 + sd)
            a = a0.copy(); p = a.positions.copy()
            for k in range(0, 640, 2):
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                p[k + 1] = p[k] + (r_max + rng.choice([-1, 1]) * 10 ** rng.uniform(lo, hi)) * u
            a.positions = p
            out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
            edge_compare(a, out, f"boundary/{base_name}_320pairs_{tagp}/seed{sd}", with_stock=True, stock_tol=1e-13)
except Exception as e:
    traceback.print_exc(); rec("boundary/wbox648_300pairs", False, err=repr(e))

# ---------------- 3. overflow detection ----------------
try:
    a = load("water648")
    ei, *_ = stock_nl(a, r_max); E = ei.shape[1]
    for Ec in (E - 1, 64):
        nl = make_nl(a, E_cap=Ec)
        out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
        f = int(out["overflow"].item())
        rec(f"overflow/E_cap={Ec}", (f & 1) == 1 and int(out["n_edges"].item()) == E, flags=f, n_edges=int(out["n_edges"].item()), E=E)
    nl = make_nl(a, E_cap=E)
    out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
    rec("overflow/E_cap=E_exact_no_flag", int(out["overflow"].item()) == 0, E=E)
    nl = make_nl(a, bin_cap=4)
    out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
    f = int(out["overflow"].item())
    rec("overflow/bin_cap=4", (f & 2) == 2, flags=f, max_bin=int(out["max_bin"].item()))
    b = load("wbox648"); nl = make_nl(b, bin_cap=8)
    out = nl.build(torch.tensor(b.positions, device=dev)); torch.cuda.synchronize()
    f = int(out["overflow"].item())
    rec("overflow/pbc_bin_cap=8", (f & 2) == 2, flags=f, max_bin=int(out["max_bin"].item()))
    # open system expanding x1.6 with few bins per dim -> bins grow, still exact (or flagged)
    nl = make_nl(a, bins_per_dim=4)
    c = a.copy(); c.positions = a.positions.mean(0) + (a.positions - a.positions.mean(0)) * 1.6
    out = nl.build(torch.tensor(c.positions, device=dev)); torch.cuda.synchronize()
    f = int(out["overflow"].item())
    if f == 0:
        edge_compare(c, out, "overflow/open_expanded_x1.6_bins4_exact")
    else:
        rec("overflow/open_expanded_x1.6_bins4_flagged", (f & 2) == 2, flags=f)
    c.positions = a.positions.mean(0) + (a.positions - a.positions.mean(0)) * 1.0
    nl = make_nl(a, bins_per_dim=2, bin_cap=64)
    out = nl.build(torch.tensor(a.positions, device=dev)); torch.cuda.synchronize()
    f = int(out["overflow"].item())
    rec("overflow/open_bins2_cap64_flagged", (f & 2) == 2, flags=f, max_bin=int(out["max_bin"].item()))
except Exception as e:
    traceback.print_exc(); rec("overflow", False, err=repr(e))

# ---------------- 4. CUDA graph capture + replay with changing positions ----------------
for name in (["water192", "wbox648_tric"] if QUICK else ["aspirin", "water648", "wbox648", "wbox648_tric", "wclus2187", "wbox2187"]):
    try:
        a0 = load(name)
        nl = make_nl(a0)
        static = torch.tensor(a0.positions, device=dev)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                nl.build(static)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = nl.build(static)
        seq = walk(a0, 5, 0.08, 7)
        if not any(a0.pbc):   # also a big rigid translation + expansion (bbox changes)
            b = a0.copy(); b.positions = (a0.positions - a0.positions.mean(0)) * 1.15 + np.array([40.0, -13.0, 7.0])
            seq.append(b)
        allok = True; Es = []
        for k, a in enumerate(seq):
            static.copy_(torch.tensor(a.positions, device=dev))
            gr.replay(); torch.cuda.synchronize()
            snap = {kk: v.clone() for kk, v in out.items()}
            ok = edge_compare(a, snap, f"cudagraph/{name}/replay{k}", with_stock=(k % 2 == 0))
            nl2 = make_nl(a0, E_cap=nl.E_cap)
            o2 = nl2.build(static); torch.cuda.synchronize()
            same = all(torch.equal(snap[kk], o2[kk]) for kk in snap)
            allok &= ok and same; Es.append(int(snap["n_edges"].item()))
        rec(f"cudagraph/{name}/replay_eq_eager", allok, E_seq=Es)
        del gr
    except Exception as e:
        traceback.print_exc(); rec(f"cudagraph/{name}", False, err=repr(e))

# ---------------- 5. whole-model check (stock e3nn eager fp64) ----------------
def model_eval(bd, stress):
    bd = dict(bd); bd["positions"] = bd["positions"].detach().clone()
    with torch_tools.default_dtype("float64"):
        o = model(bd, compute_force=True, compute_stress=stress, training=False)
    return (float(o["energy"][0]), o["forces"].detach().cpu().numpy(),
            o["stress"][0].detach().cpu().numpy() if stress and o.get("stress") is not None else None)


def stock_batch(atoms):
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    return b.to_dict()


MSYS = ["H2O", "aspirin", "org192", "water192", "org648", "water648", "wbox648", "wbox648_tric", "wbox648_outside",
        "wclus2187", "wbox2187", "single_C", "far_pair"]
if QUICK:
    MSYS = ["H2O", "aspirin", "wbox648", "single_C"]
for name in MSYS:
    try:
        a0 = load(name)
        tpl = BatchTemplate(calc, a0)
        nl = make_nl(a0)
        stress = bool(any(a0.pbc))
        for k, a in enumerate(walk(a0, 0 if QUICK else 1, 0.05, 3)):
            pos = torch.tensor(a.positions, device=dev)
            out = nl.build(pos); torch.cuda.synchronize()
            bp = pb.build(a)
            bs = stock_batch(a)
            # batch dict equality vs proto (keys/dtypes/shapes/values except edge order)
            bt = tpl.batch(out, pos, trim=True)
            keys_ok = list(bt.keys()) == list(bp.keys()) and all(bt[x].dtype == bp[x].dtype for x in bp)
            stat_ok = all(torch.equal(bt[x], bp[x]) for x in bp if x not in ("edge_index", "shifts", "unit_shifts"))
            rows = {}
            ref_p = model_eval(bp, stress); ref_s = model_eval(bs, stress)
            # floor: max |diff| among proto/stock batches and their repeats (same model) -> conditioning/atomics noise
            n = max(len(a), 1)
            ref_p2 = model_eval(bp, stress); ref_s2 = model_eval(bs, stress)   # run-to-run (atomics) noise
            def _d(x, y):
                return (abs(x[0] - y[0]) / n, float(np.abs(x[1] - y[1]).max()) if x[1].size else 0.0,
                        float(np.abs(x[2] - y[2]).max()) if stress else 0.0)
            fl = np.max([_d(ref_p, ref_s), _d(ref_p, ref_p2), _d(ref_s, ref_s2), _d(ref_p2, ref_s2)], axis=0)
            base = dict(dE_per_atom=float(fl[0]), max_dF=float(fl[1]), dstress=float(fl[2]))
            rows["floor_proto_stock_and_repeat"] = dict(base, ok=True)
            for variant in ("trim", "padded"):
                bd = tpl.batch(out, pos, trim=(variant == "trim"))
                e, f, s = model_eval(bd, stress)
                for rn, (er, fr, sr) in (("proto", ref_p), ("stock", ref_s)):
                    n = max(len(a), 1)
                    dE = abs(e - er) / n; dF = float(np.abs(f - fr).max()) if f.size else 0.0
                    dS = float(np.abs(s - sr).max()) if s is not None else 0.0
                    strict = bool(dE <= TOL_E and dF <= TOL_F and dS <= TOL_S)
                    # conditioning-limited (e.g. atoms ~3000 A outside the cell): allow 2x the proto-vs-stock floor
                    cond = bool(dE <= max(TOL_E, 2 * base["dE_per_atom"]) and dF <= max(TOL_F, 2 * base["max_dF"])
                                and dS <= max(TOL_S, 2 * base["dstress"]))
                    rows[f"{variant}_vs_{rn}"] = dict(dE_per_atom=dE, max_dF=dF, dstress=dS, ok=cond,
                                                     strict_tol=strict)
            ok = keys_ok and stat_ok and all(r["ok"] for r in rows.values())
            rec(f"model/{name}/cfg{k}", ok, keys_dtypes_ok=keys_ok, static_dyn_values_eq_proto=stat_ok,
                E=int(out["n_edges"].item()), E_cap=nl.E_cap, **{kk: vv for kk, vv in rows.items()})
    except Exception as e:
        traceback.print_exc(); rec(f"model/{name}", False, err=repr(e))

res["n_tests"] = len(res["tests"]); res["n_fail"] = len(res["failures"])
res["gpu_end"] = gpuinfo()
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(f"DONE {res['n_tests']} tests, {res['n_fail']} failures")
