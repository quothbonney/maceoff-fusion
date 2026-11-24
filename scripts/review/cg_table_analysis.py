"""CPU-only analysis of the CG value table (review item a).

For every l-triple used by the ir_config/{mace,nequip,sevennet} configs (and all triples l<=6 as a margin),
take e3nn's fp64 wigner_3j values and report:
  * the minimum gap between mathematically distinct values (after clustering at the candidate tolerance)
  * the maximum spread inside a cluster (fp64 noise between mathematically equal values)
  * tiny nonzero entries (|v| < 1e-12) that are probably mathematically zero
  * per-config unique-value counts: upstream (fp32 rounding) vs tolerance dedupe, and whether the
    256-entry uint8 limit / extcg path selection changes
  * whether fp32 rounding merges values that are distinct in fp64 (information loss upstream)
Writes results/review/cg_table_analysis.json
"""
import os, sys, json, glob, itertools
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch, e3nn.o3 as o3

R = os.environ["R"]
CFG = f"{R}/src/flashTP/ir_config"
TOL = 1e-12


def cluster(vals, tol):
    vals = sorted(vals)
    reps = []
    for v in vals:
        if reps and abs(v - reps[-1][0]) <= tol:
            reps[-1].append(v)
        else:
            reps.append([v])
    return reps


def stats(vals, tol=TOL):
    cl = cluster(vals, tol)
    spread = max(max(c) - min(c) for c in cl)
    reps = [c[0] for c in cl]
    gaps = [b - a for a, b in zip(reps, reps[1:])]
    mingap = min(gaps) if gaps else float("inf")
    i = gaps.index(mingap) if gaps else -1
    return dict(n_raw_unique=len(set(vals)), n_clusters=len(cl), max_intra_cluster_spread=spread,
                min_inter_cluster_gap=mingap, closest_pair=(reps[i], reps[i + 1]) if gaps else None)


def triple_values(l1, l2, l3):
    c = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
    nz = c[c != 0]
    tiny = int(((c.abs() < 1e-12) & (c != 0)).sum())
    return nz.tolist(), tiny


def load_cfg(fn):
    out = []
    for line in open(fn).read().split("\n"):
        if not line.strip():
            continue
        tp = json.loads(line)["tp"]
        i1, i2, io = o3.Irreps(tp[0]), o3.Irreps(tp[1]), o3.Irreps(tp[2])
        out.append([(i1[a].ir.l, i2[b].ir.l, io[c].ir.l) for a, b, c, *_ in tp[3]])
    return out


res = {"tol": TOL, "global": {}, "configs": {}}
# all triples up to lmax
for LM in (3, 5, 6):
    vals, tiny = [], 0
    for l1, l2, l3 in itertools.product(range(LM + 1), repeat=3):
        if abs(l1 - l2) <= l3 <= l1 + l2:
            v, t = triple_values(l1, l2, l3); vals += v; tiny += t
    s = stats(vals); s["tiny_nonzero_entries"] = tiny
    # fp32 rounding information loss: distinct fp64 clusters that collapse to the same fp32 value
    reps = [c[0] for c in cluster(vals, TOL)]
    f32 = torch.tensor(reps, dtype=torch.float64).float()
    s["fp32_collapsed_distinct_values"] = len(reps) - len(set(f32.tolist()))
    s["max_abs_fp32_rounding_err"] = float((f32.double() - torch.tensor(reps, dtype=torch.float64)).abs().max())
    # tolerance sensitivity: number of clusters across tolerances
    s["clusters_vs_tol"] = {f"{t:g}": len(cluster(vals, t)) for t in (1e-15, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6)}
    res["global"][f"lmax{LM}_all_triples"] = s
    print(f"lmax={LM}", json.dumps(s))

for fn in sorted(glob.glob(f"{CFG}/*/*")):
    name = os.path.relpath(fn, CFG)
    for li, triples in enumerate(load_cfg(fn)):
        vals = []
        for t in triples:
            vals += o3.wigner_3j(*t, dtype=torch.float64).unique().tolist()
        up = len(set(torch.tensor(vals, dtype=torch.float64).float().double().tolist()))  # upstream count
        s = stats(vals)
        s["n_upstream_fp32_rounded"] = up
        s["path_upstream"] = "large" if up <= 256 else "extcg"
        s["path_tol_dedupe"] = "large" if s["n_clusters"] <= 256 else "extcg"
        s["lmax"] = max(max(t) for t in triples)
        del s["closest_pair"]
        res["configs"][f"{name}:L{li}"] = s
        print(name, li, json.dumps(s))

json.dump(res, open(f"{R}/results/review/cg_table_analysis.json", "w"), indent=1)
