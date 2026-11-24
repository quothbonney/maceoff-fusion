"""Whole-model parity vs e3nn: hybrid, cueq-only (enable_cueq=True), FlashTP-fp64CG-only. Usage: model_parity.py out.json"""
import json, sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from mace.calculators import MACECalculator
from mace.data.neighborhood import get_neighborhood
out_path = sys.argv[1]; NREP = 5; NSEQ = 5
calcs = {"e3nn": HB.build_calc(MODEL, conv="e3nn", sc="e3nn")[0],
         "hybrid": HB.build_calc(MODEL)[0],
         "flashtp_fp64cg_only": HB.build_calc(MODEL, conv="flashtp", sc="e3nn")[0],
         "cueq_only": MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False, enable_cueq=True)}
rmax = float(calcs["e3nn"].models[0].r_max)
res = dict(env=env_info(), r_max=rmax, nrep=NREP, nseq=NSEQ, describe={k: HB.describe(c.models[0]) for k, c in calcs.items()}, systems={})

def ef(calc, atoms, pos=None):
    a = atoms.copy()
    if pos is not None: a.positions = pos
    a.calc = calc; calc.reset()
    return a.get_potential_energy(), a.get_forces().copy()

def cmp(ref, t, nat):
    dF = np.abs(t[1] - ref[1])
    return dict(dE_per_atom=abs(t[0] - ref[0]) / nat, max_dF=float(dF.max()), rms_dF=float(np.sqrt(((t[1] - ref[1]) ** 2).mean())))

for name in ["H2O", "water192", "water648", "water1536"]:
    atoms = get_system(name); nat = len(atoms)
    # changing-coordinate sequence: cumulative 0.02 A gaussian steps, seed 100+k
    seq, p = [], atoms.positions.copy()
    for k in range(NSEQ):
        p = p + np.random.default_rng(100 + k).normal(scale=0.02, size=p.shape); seq.append(p.copy())
    edges = []
    for q in [atoms.positions] + seq:
        ei = get_neighborhood(q, cutoff=rmax, pbc=(False,) * 3, cell=np.zeros((3, 3)))[0]
        edges.append(set(map(tuple, ei.T.tolist())))
    nl = dict(n_edges=[len(e) for e in edges], edges_changed_vs_prev=[len(edges[i] ^ edges[i - 1]) for i in range(1, len(edges))])
    out = {}
    for k, c in calcs.items():
        rep = [ef(c, atoms) for _ in range(NREP)]
        sq = [ef(c, atoms, q) for q in seq]
        out[k] = dict(rep=rep, seq=sq)
    ref = out["e3nn"]
    d = dict(n_atoms=nat, neighbour_list=nl, E_ref=ref["rep"][0][0], max_abs_F_ref=float(np.abs(ref["rep"][0][1]).max()), variants={})
    for k, o in out.items():
        fixed = [cmp(ref["rep"][0], r, nat) for r in o["rep"]]
        seqc = [cmp(r, t, nat) for r, t in zip(ref["seq"], o["seq"])]
        d["variants"][k] = dict(
            fixed=dict(dE_per_atom=max(x["dE_per_atom"] for x in fixed), max_dF=max(x["max_dF"] for x in fixed), rms_dF=max(x["rms_dF"] for x in fixed)),
            sequence=dict(dE_per_atom=max(x["dE_per_atom"] for x in seqc), max_dF=max(x["max_dF"] for x in seqc), rms_dF=max(x["rms_dF"] for x in seqc), per_step=seqc),
            repeat=dict(max_dE=float(max(abs(r[0] - o["rep"][0][0]) for r in o["rep"])),
                        max_dF=float(max(np.abs(r[1] - o["rep"][0][1]).max() for r in o["rep"])),
                        bitwise=all(r[0] == o["rep"][0][0] and np.array_equal(r[1], o["rep"][0][1]) for r in o["rep"])),
            finite=bool(all(np.isfinite(e) and np.isfinite(f).all() for e, f in o["rep"] + o["seq"])))
    res["systems"][name] = d
    print(name, json.dumps({k: {q: v[q] for q in ("fixed", "repeat")} | {"seq": {z: v["sequence"][z] for z in ("dE_per_atom", "max_dF", "rms_dF")}} for k, v in d["variants"].items()}, indent=1), nl, flush=True)
with open(out_path, "w") as f: json.dump(res, f, indent=2, default=float)
