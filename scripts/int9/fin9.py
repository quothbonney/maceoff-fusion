"""Merge final timing runs (int9 B1/B2 + md9 B3) into results/int9/INTEGRATION.json and print markdown tables.
usage: python3 fin9.py final_run1.json final_run2.json [final_run3_wbox192.json] -- B3 from results/md9/b3_*.json"""
import json, sys, glob, os
R = "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study"
runs = [json.load(open(p)) for p in sys.argv[1:]]
SYS = ["aspirin", "water192", "wbox192", "org648", "water648", "wbox648", "water2187", "wbox2187", "water5184", "wbox5184"]
CF = [("stock", "stock"), ("hybrid+embed+node", "FlashTP-hybrid"), ("nodehybrid+embed+node", "node-hybrid (P8)"),
      ("gpustep", "GPUStep (P9)"), ("gpustep_hyb", "GPUStep w/ FlashTP model")]


def get(s, c, k, run):
    D = run["systems"].get(s, {}).get(c)
    if not D or k not in D: return None
    return D[k]


def pick(s, c, k):
    """per run: (run index, value). wbox192 gpustep: run1 used the brute stub (superseded) -> use later runs only."""
    out = []
    for i, r in enumerate(runs):
        if False:
            continue
        if s not in r["systems"]: continue
        v = get(s, c, k, r)
        if v is not None: out.append(v)
    return out


def fmt(vs):
    if not vs: return "-"
    m = vs[0]
    t = f"{m['median']:.3f} [{m['p10']:.3f}-{m['p90']:.3f}]"
    if len(vs) > 1: t += f" / {vs[1]['median']:.3f}"
    return t


# B3 from md9: newest b3 file containing (provider, system)
b3 = {}
for f in sorted(glob.glob(f"{R}/results/md9/b3_*.json"), key=os.path.getmtime) + [f"{R}/results/int9/b3_final_gpustep.json"]:
    d = json.load(open(f))
    for r in d.get("rows", []):
        if "ms_step_median" in r:
            b3[(r["provider"], r["system"])] = dict(ms=r["ms_step_median"], blocks=r.get("ms_step_blocks"),
                                                    aps=r.get("atom_steps_per_s"), job=d.get("slurm_job"),
                                                    syncs=r.get("explicit_syncs_per_step"), name=r.get("provider_name"))
out = dict(runs=[dict(job=r["job"], node=r["node"], gpu=r["gpu_start"]) for r in runs], table={})
print("| system | N | E | backend | B1 ms runA [p10-p90] / runB | B2 ms runA [p10-p90] / runB | B3 ms/step (stock: ASE VV; others: b3bench job 1147) | B2 speedup vs node-hybrid / FlashTP-hybrid / stock | peak MiB |")
print("|---|---|---|---|---|---|---|---|---|")
for s in SYS:
    N = next((r["systems"][s]["N"] for r in runs if s in r["systems"]), None)
    E = next((get(s, "nodehybrid+embed+node", "E", r) for r in runs if s in r["systems"]), None)
    b2 = {c: pick(s, c, "B2") for c, _ in CF}
    for c, lab in CF:
        b1v, b2v = pick(s, c, "B1"), b2[c]
        if not b2v: continue
        b3k = {"stock": "ase", "nodehybrid+embed+node": "graphed", "gpustep": "gpustep"}.get(c)
        b3v = b3.get((b3k, s)) if b3k else None
        b3s = f"{b3v['ms']:.3f} (job {b3v['job']})" if b3v else "-"
        if c == "stock" and b3.get(("stock", s)):
            b3s += f"; VV+stock {b3[('stock', s)]['ms']:.2f}"
        sp = ""
        if c.startswith("gpustep"):
            m = b2v[0]["median"]
            sp = " / ".join(f"{b2[o][0]['median'] / m:.2f}x" if b2[o] else "-" for o in ("nodehybrid+embed+node", "hybrid+embed+node", "stock"))
        pk = pick(s, c, "peak_MiB")
        print(f"| {s} | {N} | {E} | {lab} | {fmt(b1v)} | {fmt(b2v)} | {b3s} | {sp} | {pk[0]:.0f} |" if pk else "")
        out["table"][f"{s}/{c}"] = dict(N=N, E=E, B1=b1v, B2=b2v, B3=b3v, peak_MiB=pk[:1],
                                        first_step_s=pick(s, c, "first_step_s")[:1], E_cap=pick(s, c, "E_cap")[:1],
                                        B2chk=pick(s, c, "B2chk"), path=pick(s, c, "path")[:1])
json.dump(out, open(f"{R}/results/int9/INTEGRATION.json", "w"), indent=1, default=str)
