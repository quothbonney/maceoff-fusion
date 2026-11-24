"""CPU analysis for md9 -> results/md9/summary9.json + markdown tables on stdout (pasted into MD9.md).
usage: analyze9.py [b3 json, ...]"""
import os, sys, json, glob
import numpy as np
R = os.environ["R"]; OUT = f"{R}/results/md9"
S = dict(dyn=[], drift={}, vv_vs_ase={}, b3=[])


def e(x):
    return "-" if x is None else f"{x:.1e}"


# ---------------- dynamic correctness
print("## Dynamic correctness\n")
print("| provider | system | N | pbc | steps (dt fs) | samples | worst dE/atom eV | worst max|dF| eV/A | worst RMS dF | worst dstress eV/A^3 | max|F| | edges == stock (prov / proto) | E range (directed) | edges in / out | syncs/step | rollbacks | fails |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{OUT}/dyn_*.json")):
    if "smoke" in f:
        continue
    d = json.load(open(f))
    if "negctrl" in f:
        d["provider"] = d["provider"] + " **NEG-CTRL drop_edge**"
    w = d["worst"]
    ps = d.get("provider_syncs", {})
    sps = {k: round(v / max(d["steps"], 1), 3) for k, v in ps.items()}
    row = dict(file=os.path.basename(f), provider=d["provider"], system=d["system"], N=d["N"], periodic=d["periodic"],
               steps=d["steps"], dt=d["dt_fs"], samples=d["n_samples"], fails=d["n_fail"], worst=w,
               edges_prov_eq=d["edges_all_equal_stock"] if d["provider_edges_checked"] else None,
               proto_eq=d["proto_all_equal_stock"], edges=(d["edges_min"], d["edges_max"]),
               entered=d["pairs_entered_total"], left=d["pairs_left_total"], syncs_per_step=sps,
               rollbacks=d.get("n_rollbacks"), job=d["slurm_job"], iso=d.get("isolated_atom"),
               check_events=len(d.get("check_failures") or []))
    S["dyn"].append(row)
    print(f"| {d['provider']} | {d['system']} | {d['N']} | {'T' if d['periodic'] else 'F'} | {d['steps']} ({d['dt_fs']}) | "
          f"{d['n_samples']} | {e(w['dE_per_atom'])} | {e(w['max_dF'])} | {e(w['rms_dF'])} | "
          f"{e(w['dstress']) if d['periodic'] else 'n/a'} | {w['maxF']:.1f} | {row['edges_prov_eq']} / {row['proto_eq']} | "
          f"{d['edges_min']}-{d['edges_max']} | {row['entered']} / {row['left']} | {sps} | {row['rollbacks']} | {d['n_fail']} |")
for r in S["dyn"]:
    if r["iso"]:
        print(f"\nisolated atom ({r['provider']}): {r['iso']}")

# ---------------- drift
P4 = json.load(open(f"{R}/results/md/summary.json"))["energy"]
print("\n## NVE drift (per atom)\n")
print("| run | provider | system | dt fs | steps | t ps | drift eV/atom/ps | block std | RMS fluct eV/atom | max-min | T start/end/mean K | ms/step (incl. snapshots) | job |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{OUT}/traj_*.json")):
    d = json.load(open(f)); en = d["energy"]; tag = os.path.basename(f)[5:-5]
    S["drift"][tag] = dict(provider=d["provider"], system=d["system"], energy=en, ms_per_step=d["ms_per_step"],
                           job=d["slurm_job"], syncs=d.get("provider_syncs"), check_events=d.get("check_failures"))
    print(f"| {tag} | {d['provider']} | {d['system']} | {en['dt_fs']} | {en['n_steps']} | {en['t_ps']:.2f} | "
          f"{en['drift_eV_atom_ps']:.2e} | {en['block_slope_std']:.1e} | {en['rms_fluct_eV_atom']:.2e} | "
          f"{en['max_minus_min_eV_atom']:.1e} | {en['T_start_K']:.0f}/{en['T_end_K']:.0f}/{en['T_mean_K']:.0f} | "
          f"{d['ms_per_step']:.2f} | {d['slurm_job']} |")
for k in ("e3nn_dt0.5", "e3nn_dt0.5_rerun", "flash_dt0.5", "e3nn_dt0.25", "w648_e3nn_dt0.5"):
    v = P4[k]
    print(f"| Phase4 {k} (ASE) | | | {v['dt_fs']} | {v['n_steps']} | {v['t_ps']:.2f} | {v['drift_eV_atom_ps']:.2e} | "
          f"{v['block_slope_std']:.1e} | {v['rms_fluct_eV_atom']:.2e} | {v['max_minus_min_eV_atom']:.1e} | "
          f"{v['T_start_K']:.0f}/{v['T_end_K']:.0f}/- | | |")

# ---------------- neighbour flips along drift runs (stock-equivalent pair counting, ASE neighbour_list)
try:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    print("\n## Neighbour-list changes along drift runs (pairs d < 5 A incl. periodic image; saved frames)\n")
    for f in sorted(glob.glob(f"{OUT}/traj_*.npz")):
        tr = np.load(f); tag = os.path.basename(f)[5:-4]
        prev = None; flips = 0; ne = []
        for P in tr["positions"]:
            a = Atoms(numbers=tr["numbers"], positions=P, cell=tr["cell"], pbc=tr["pbc"])
            i, j, Sx = neighbor_list("ijS", a, 5.0)
            cur = set(zip(i.tolist(), j.tolist(), *Sx.T.tolist())); ne.append(len(cur))
            if prev is not None:
                flips += len(cur ^ prev)
            prev = cur
        S["drift"][tag]["nlist"] = dict(directed_edges_min=min(ne), directed_edges_max=max(ne),
                                        directed_edge_flips_between_saved_frames=flips, frames=len(ne),
                                        save_every=int(tr["saved_steps"][1] - tr["saved_steps"][0]))
        print(f"- {tag}: directed edges {min(ne)}-{max(ne)}, flips between consecutive saved frames "
              f"(every {S['drift'][tag]['nlist']['save_every']} steps) = {flips}")
except Exception as ex:
    print("nlist analysis failed", ex)

# ---------------- VV(GPU)+stock vs Phase-4 ASE VelocityVerlet + MACECalculator (same start, same dt)
pa, pb = f"{OUT}/traj_stock_water192_dt0.5.npz", f"{R}/results/md/traj_e3nn_dt0.5.npz"
if os.path.exists(pa):
    a, b = np.load(pa), np.load(pb)
    m = min(len(a["positions"]), len(b["positions"]))
    assert np.array_equal(a["saved_steps"][:m], b["saved_steps"][:m])
    d = a["positions"][:m] - b["positions"][:m]; rmsd = np.sqrt((d ** 2).sum(-1).mean(-1))
    n = min(len(a["epot"]), len(b["epot"]))
    dep = np.abs(a["epot"][:n] - b["epot"][:n]) / 192
    t = a["saved_steps"][:m] * 0.5 / 1000
    pick = {}
    for tt in (0.005, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
        k = np.argmin(np.abs(t - tt)); pick[tt] = float(rmsd[k])
    msk = (rmsd > 1e-13) & (rmsd < 1e-3)
    lam = float(np.polyfit(t[msk], np.log(rmsd[msk]), 1)[0]) if msk.sum() > 5 else None
    ctrl = json.load(open(f"{R}/results/md/summary.json"))["divergence"].get("e3nn_dt0.5|e3nn_dt0.5_rerun", {})
    cr = dict((round(x, 4), y) for x, y in ctrl.get("rmsd_A", []))
    S["vv_vs_ase"] = dict(rmsd_A_at_ps=pick, first_frame_rmsd=float(rmsd[1]), step0_dEpot_per_atom=float(dep[0]),
                          dEpot_per_atom_first10=[float(x) for x in dep[:10]], lyapunov_per_ps=lam,
                          phase4_e3nn_rerun_rmsd_at_ps={k: cr.get(k) for k in pick},
                          phase4_rerun_lambda=ctrl.get("lyapunov_fit_per_ps"))
    print("\n## GPU VV (stock provider) vs Phase-4 ASE VelocityVerlet (water192, dt 0.5, same start)\n")
    print(json.dumps(S["vv_vs_ase"], indent=1))

# ---------------- B3
files = sys.argv[1:] or sorted(f for f in glob.glob(f"{OUT}/b3_*.json") if "smoke" not in f)
print("\n## B3 sustained MD throughput (dt 0.5 fs; median of 4 blocks [min-max], ms/step)\n")
print("| system | N | pbc | provider | ms/step | atom-steps/s | ns/day | sync-audit syncs/step | explicit | peak MiB | setup s | job |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in files:
    d = json.load(open(f))
    for r in d["rows"]:
        r = dict(r); r["job"] = d["slurm_job"]; S["b3"].append(r)
        if "error" in r:
            print(f"| {r['system']} | | | {r['provider']} | ERROR {r['error'][:80]} | | | | | | | {d['slurm_job']} |"); continue
        au = r["sync_audit"]["per_step"] if isinstance(r["sync_audit"], dict) else r["sync_audit"]
        ex = {k: round(v, 3) for k, v in (r.get("explicit_syncs_per_step") or {}).items()}
        print(f"| {r['system']} | {r['N']} | {'T' if r['periodic'] else 'F'} | {r['provider']} | {r['ms_step_median']:.3f} "
              f"[{r['ms_step_min']:.3f}-{r['ms_step_max']:.3f}] | {r['atom_steps_per_s']:.3e} | {r['ns_per_day_at_dt']:.2f} | "
              f"{au} | {ex} | {r['peak_mem_MiB']:.0f} | {r['setup_first_force_s']:.2f} | {d['slurm_job']} |")
json.dump(S, open(f"{OUT}/summary9.json", "w"), indent=1, default=float)
