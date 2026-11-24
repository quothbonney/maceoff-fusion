"""md9 dynamic correctness harness.
usage: dyncheck.py <provider stock|graphed|gpustep> <system[,system...]> [--steps 500] [--every 5] [--dt 0.5]
Runs NVE velocity Verlet (vv.VelocityVerletGPU) with the provider under test from results/md9/systems/start_<sys>.npz.
Every `every` steps (incl. step 0) the identical coordinates (exact copy of the GPU positions the provider just
evaluated) are re-evaluated with stock fp64 MACE eager (fresh stock batch, matscipy NL) and compared:
  |dE|/atom, max|dF|, RMS|dF|, stress (periodic; provider runs with stress), neighbour sets: provider edge set
  (sorted (sender, receiver, unit_shift) rows) vs stock and vs proto_graph builder (bitwise array equality),
  edge counts, pairs entering/leaving the cutoff between consecutive samples (stock list).
Tolerances (results/phase9/INTERFACES.md): dE/atom 1e-11 eV, max|dF| 1e-11 eV/A, stress 1e-14 eV/A^3, edges equal.
Output results/md9/dyn_<provider>_<system>.json"""
import os, sys, json, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vv as V  # first
import numpy as np
import torch
from proto_graph import ProtoBuilder

TOL = dict(dE_per_atom=1e-11, max_dF=1e-11, dstress=1e-14)
ap = argparse.ArgumentParser()
ap.add_argument("provider"); ap.add_argument("systems")
ap.add_argument("--steps", type=int, default=500); ap.add_argument("--every", type=int, default=5)
ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--tag", default="")
ap.add_argument("--check_every", type=int, default=10)
ap.add_argument("--inject", default="", help="negative control: 'drop_edge' removes the last directed edge in the graphed "
                "provider's builder output at every step")
args = ap.parse_args()


def rows_of(ei, us):
    a = np.concatenate([np.asarray(ei).T.astype(np.int64), np.rint(np.asarray(us)).astype(np.int64)], 1)
    assert np.array_equal(np.rint(us), us), "non-integer unit shifts"
    return a[np.lexsort(a.T[::-1])]


def pairset(rows):
    return set(map(tuple, rows.tolist()))


_rc = V.stock_calc()


def run(sysname):
    atoms = V.load_start(sysname)
    N = len(atoms); periodic = any(atoms.pbc)
    reff = V.StockEager(atoms, stress=periodic, calc=_rc, keep_edges=True)
    pb = ProtoBuilder(_rc)
    t0 = time.time()
    prov = V.make_provider(args.provider, atoms, stress=periodic, keep_edges=True,
                           **({} if args.provider == "gpustep" else {}))
    t_build = time.time() - t0
    if args.inject == "drop_edge":
        b0 = prov.runner._build

        def bad(a):
            bd = dict(b0(a))
            for k in ("edge_index",):
                bd[k] = bd[k][:, :-1]
            for k in ("shifts", "unit_shifts"):
                bd[k] = bd[k][:-1]
            prov._bd = bd
            return bd
        prov.runner._build = bad
    md = V.VelocityVerletGPU(prov, atoms, args.dt, args.steps, check_every=args.check_every if args.provider == "gpustep" else 0)
    samples, prev = [], None
    entered = left = 0
    fails = []
    t1 = time.time()
    for s in range(0, args.steps + 1):
        if s > 0:
            md.step()
        if s % args.every:
            continue
        torch.cuda.synchronize()
        x = md.x.detach().cpu().numpy().copy()
        Ep = float(md.epot[s].item()); Fp = md.F.detach().cpu().numpy().copy()
        Sp = V.voigt(prov.last_stress) if (periodic and prov.last_stress is not None) else None
        Er, Fr = reff.eval_np(x)
        Er = float(Er); Fr = Fr.cpu().numpy(); Sr = V.voigt(reff.last_stress) if periodic else None
        r_rows = rows_of(*reff.edges())
        bd = pb.build(reff.atoms); p_rows = rows_of(bd["edge_index"].cpu().numpy(), bd["unit_shifts"].cpu().numpy())
        pe = prov.edges() if hasattr(prov, "edges") else None
        v_rows = rows_of(*pe) if pe is not None else None
        dF = np.abs(Fp - Fr)
        row = dict(step=s, E_ref_edges=len(r_rows), dE_per_atom=abs(Ep - Er) / N, max_dF=float(dF.max()),
                   rms_dF=float(np.sqrt(((Fp - Fr) ** 2).sum(-1).mean())), maxF=float(np.abs(Fr).max()),
                   proto_eq_stock=bool(np.array_equal(p_rows, r_rows)),
                   prov_edges=None if v_rows is None else len(v_rows),
                   prov_eq_stock=None if v_rows is None else bool(np.array_equal(v_rows, r_rows)),
                   prov_eq_proto=None if v_rows is None else bool(np.array_equal(v_rows, p_rows)))
        if Sp is not None:
            row["dstress"] = float(np.abs(Sp - Sr).max()); row["stress_ref_max"] = float(np.abs(Sr).max())
        cur = pairset(r_rows)
        if prev is not None:
            row["entered"] = len(cur - prev); row["left"] = len(prev - cur)
            entered += row["entered"]; left += row["left"]
        prev = cur
        ok = (row["dE_per_atom"] <= TOL["dE_per_atom"] and row["max_dF"] <= TOL["max_dF"]
              and row.get("dstress", 0.0) <= TOL["dstress"] and row["proto_eq_stock"]
              and row["prov_eq_stock"] in (None, True))
        if periodic and Sp is None:
            ok = False; row["note"] = "provider gave no stress"
        row["ok"] = bool(ok)
        if not ok:
            fails.append(row)
        samples.append(row)
        if s % (args.every * 20) == 0:
            print(f"{sysname} {prov.name} step {s} E={len(r_rows)} dE/at={row['dE_per_atom']:.2e} "
                  f"maxdF={row['max_dF']:.2e} dS={row.get('dstress', 0):.2e} eq={row['prov_eq_stock']} ok={ok}", flush=True)
    wall = time.time() - t1
    ep, ek = md.energies(); et = (ep + ek) / N
    out = dict(V.env_info(), provider=prov.name, system=sysname, N=N, periodic=periodic, dt_fs=args.dt,
               steps=args.steps, every=args.every, tol=TOL, build_s=t_build, wall_s=wall,
               n_samples=len(samples), n_fail=len(fails),
               worst=dict(dE_per_atom=max(r["dE_per_atom"] for r in samples), max_dF=max(r["max_dF"] for r in samples),
                          rms_dF=max(r["rms_dF"] for r in samples),
                          dstress=max([r.get("dstress", 0.0) for r in samples]),
                          maxF=max(r["maxF"] for r in samples)),
               edges_all_equal_stock=all(r["prov_eq_stock"] in (None, True) for r in samples),
               provider_edges_checked=samples[0]["prov_eq_stock"] is not None,
               proto_all_equal_stock=all(r["proto_eq_stock"] for r in samples),
               edges_min=min(r["E_ref_edges"] for r in samples), edges_max=max(r["E_ref_edges"] for r in samples),
               pairs_entered_total=entered, pairs_left_total=left,
               Etot_per_atom_range=float(et.max() - et.min()), T_start_K=float(2 * ek[0] / (3 * N * V.units.kB)),
               T_end_K=float(2 * ek[-1] / (3 * N * V.units.kB)),
               provider_syncs=dict(prov.syncs), check_failures=md.check_fail, n_rollbacks=md.n_rollbacks,
               runner_stats=prov.runner.stats() if hasattr(prov, "runner") else
               (prov.sim.stats() if hasattr(prov, "sim") and hasattr(prov.sim, "stats") else None),
               samples=samples, failures=fails[:20])
    if sysname == "water192_iso":
        iso = N - 1
        out["isolated_atom"] = dict(index=iso, max_abs_F_last=float(np.abs(Fp[iso]).max()),
                                    ref_max_abs_F_last=float(np.abs(Fr[iso]).max()),
                                    edges_touching=int(((r_rows[:, 0] == iso) | (r_rows[:, 1] == iso)).sum()))
    tag = f"_{args.tag}" if args.tag else ""
    json.dump(out, open(f"{V.R}/results/md9/dyn_{args.provider}_{sysname}{tag}.json", "w"), indent=1, default=float)
    print(f"DONE {sysname} {prov.name} samples={len(samples)} fails={len(fails)} worst={out['worst']} "
          f"edges_eq={out['edges_all_equal_stock']} +{entered}/-{left} wall={wall:.1f}s", flush=True)
    del md, prov
    torch.cuda.empty_cache()


for sname in args.systems.split(","):
    try:
        run(sname)
    except Exception:
        traceback.print_exc()
        print(f"ERROR {sname}", flush=True)
