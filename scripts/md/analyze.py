"""CPU analysis of results/md/traj_*.npz -> results/md/summary.json (+ printed tables)."""
import os, sys, json, glob
import numpy as np
from scipy.spatial import cKDTree
R = os.environ["R"]; OUT = f"{R}/results/md"; RMAX = 5.0
def load(tag):
    p = f"{OUT}/traj_{tag}.npz"
    return np.load(p) if os.path.exists(p) else None
def energy_stats(tr):
    n = int(tr["n_atoms"]); dt = float(tr["dt_fs"]); et = (tr["epot"] + tr["ekin"]) / n
    t = np.arange(len(et)) * dt / 1000.0  # ps
    slope, icpt = np.polyfit(t, et, 1); resid = et - (slope * t + icpt)
    nb = 4; blk = np.array_split(np.arange(len(et)), nb)
    bslopes = [np.polyfit(t[b], et[b], 1)[0] for b in blk]
    # fluctuation over common first 1 ps window, about its own linear fit
    w = t <= 1.0; s1, i1 = np.polyfit(t[w], et[w], 1)
    return dict(dt_fs=dt, n_steps=len(et) - 1, t_ps=float(t[-1]), drift_eV_atom_ps=slope,
                block_slopes_eV_atom_ps=bslopes, block_slope_std=float(np.std(bslopes, ddof=1)),
                rms_fluct_eV_atom=float(resid.std()), rms_fluct_first1ps_eV_atom=float((et[w] - (s1 * t[w] + i1)).std()),
                std_raw_eV_atom=float(et.std()), max_minus_min_eV_atom=float(et.max() - et.min()),
                Etot0_eV_atom=float(et[0]), Etot_end_minus_0=float(et[-1] - et[0]),
                T_start_K=float(2 * tr["ekin"][0] / (3 * n * 8.617333262e-5)), T_end_K=float(2 * tr["ekin"][-1] / (3 * n * 8.617333262e-5)))
def pairs(pos):
    return set(map(tuple, cKDTree(pos).query_pairs(RMAX, output_type="ndarray")))
def nlist(tr, stride=1):
    P = tr["positions"]; p0 = pairs(P[0]); out = []
    for i in range(0, len(P), stride):
        p = pairs(P[i]); out.append([int(tr["saved_steps"][i]), 2 * len(p), len(p - p0), len(p0 - p)])
    return out
def rmsd(a, b):
    m = min(len(a["positions"]), len(b["positions"]))
    d = a["positions"][:m] - b["positions"][:m]
    return a["saved_steps"][:m], np.sqrt((d ** 2).sum(-1).mean(-1)), d
summ = dict(energy={}, nlist={}, divergence={})
tags = sorted(os.path.basename(p)[5:-4] for p in glob.glob(f"{OUT}/traj_*.npz") if "smoke" not in p)
for tag in tags:
    tr = load(tag); summ["energy"][tag] = energy_stats(tr)
    nl = nlist(tr); summ["nlist"][tag] = dict(rows=nl, edges_start=nl[0][1], edges_end=nl[-1][1],
        edges_min=min(r[1] for r in nl), edges_max=max(r[1] for r in nl), pairs_added_end=nl[-1][2], pairs_removed_end=nl[-1][3],
        frames_with_changed_set=int(sum(1 for r in nl if r[2] + r[3] > 0)), n_frames=len(nl),
        changes_between_consecutive_frames_total=None)
    P = tr["positions"]; tot = 0
    for i in range(1, len(P)):
        a, b = pairs(P[i - 1]), pairs(P[i]); tot += len(a ^ b)
    summ["nlist"][tag]["changes_between_consecutive_frames_total"] = tot
for x, y in [("e3nn_dt0.5", "flash_dt0.5"), ("e3nn_dt0.5", "e3nn_dt0.5_rerun"), ("flash_dt0.5", "flash_dt0.5_rerun"),
             ("e3nn_dt0.25", "flash_dt0.25"), ("e3nn_dt1.0", "flash_dt1.0"), ("w648_e3nn_dt0.5", "w648_flash_dt0.5")]:
    a, b = load(x), load(y)
    if a is None or b is None: continue
    steps, r, d = rmsd(a, b); dt = float(a["dt_fs"])
    nz = np.nonzero(r)[0]
    summ["divergence"][f"{x}|{y}"] = dict(bitwise_identical_positions=bool((r == 0).all()),
        first_nonzero_step=int(steps[nz[0]]) if len(nz) else None,
        maxabs_dx_first_nonzero=float(np.abs(d[nz[0]]).max()) if len(nz) else 0.0,
        epot_bitwise_identical=bool(np.array_equal(a["epot"][:min(len(a["epot"]), len(b["epot"]))], b["epot"][:min(len(a["epot"]), len(b["epot"]))])),
        rmsd_A=[[float(s * dt / 1000), float(v)] for s, v in zip(steps, r)])
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print("ENERGY (per atom)"); print(f"{'tag':22s} {'dt':>5s} {'t_ps':>5s} {'drift eV/at/ps':>15s} {'blk std':>9s} {'rms fl':>9s} {'rms fl 1ps':>10s} {'T0':>6s} {'Tend':>6s}")
for k, v in summ["energy"].items():
    print(f"{k:22s} {v['dt_fs']:5.2f} {v['t_ps']:5.2f} {v['drift_eV_atom_ps']:15.3e} {v['block_slope_std']:9.2e} {v['rms_fluct_eV_atom']:9.2e} {v['rms_fluct_first1ps_eV_atom']:10.2e} {v['T_start_K']:6.1f} {v['T_end_K']:6.1f}")
print("NLIST (directed edges r<5.0 A)")
for k, v in summ["nlist"].items():
    print(f"{k:22s} start={v['edges_start']} end={v['edges_end']} min={v['edges_min']} max={v['edges_max']} +pairs={v['pairs_added_end']} -pairs={v['pairs_removed_end']} frames_changed={v['frames_with_changed_set']}/{v['n_frames']} consecutive_flips={v['changes_between_consecutive_frames_total']}")
print("DIVERGENCE (position RMSD, A)")
for k, v in summ["divergence"].items():
    r = dict((round(t, 4), x) for t, x in v["rmsd_A"])
    pick = [t for t in [0.005, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0] if t in r]
    print(k, "bitwise=", v["bitwise_identical_positions"], "first_nonzero_step=", v["first_nonzero_step"], v["maxabs_dx_first_nonzero"], " ".join(f"{t}ps:{r[t]:.2e}" for t in pick))
print("LYAPUNOV fit: ln RMSD vs t over RMSD in [1e-13, 1e-3] A")
for k, v in summ["divergence"].items():
    a = np.array(v["rmsd_A"]); m = (a[:, 1] > 1e-13) & (a[:, 1] < 1e-3)
    if m.sum() > 5:
        lam = np.polyfit(a[m, 0], np.log(a[m, 1]), 1)[0]; v["lyapunov_fit_per_ps"] = lam
        print(f"  {k}: lambda={lam:.1f} /ps (e-fold {1000/lam:.1f} fs), window {a[m,0].min():.3f}-{a[m,0].max():.3f} ps")
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
