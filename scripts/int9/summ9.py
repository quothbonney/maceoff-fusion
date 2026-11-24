"""Render bench9 JSON(s) as markdown tables. usage: python summ9.py bench.json [bench2.json ...]"""
import json, sys

ORDER = ["aspirin", "H2O", "water192", "wbox192", "org648", "water648", "wbox648", "water2187", "wbox2187", "water5184", "wbox5184"]


def f(x):
    if not isinstance(x, dict) or "median" not in x:
        return "-"
    return f"{x['median']:.3f} [{x['p10']:.3f}-{x['p90']:.3f}]"


for path in sys.argv[1:]:
    d = json.load(open(path))
    cfgs = list(d["configs"].keys())
    print(f"\n### {path}  (job {d['job']}, {d['node']}, {d['gpu_start'].split(',')[2].strip()}; warm {d['warm']} reps {d['reps']})\n")
    print("| system | N | E -> E_cap | config | B1 ms | B2 ms | B2 other | NL only ms | peak MiB | first step s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(d["systems"], key=lambda k: ORDER.index(k) if k in ORDER else 99):
        S = d["systems"][s]
        for c in cfgs:
            D = S.get(c)
            if D is None:
                continue
            if "error" in D:
                print(f"| {s} | {S['N']} | - | {c} | ERROR {D['error'][:80]} | | | | | |"); continue
            other = ""
            if "B2host" in D: other = f"host-proto {D['B2host']['median']:.3f}"
            if "B2chk" in D: other = f"+check/step {D['B2chk']['median']:.3f}"
            nlo = f"{D['B2_nl_only']['median']:.3f}" if "B2_nl_only" in D else "-"
            ec = f"{D.get('E')} -> {D.get('E_cap', '-')}"
            print(f"| {s} | {S['N']} | {ec} | {c} | {f(D.get('B1'))} | {f(D.get('B2'))} | {other} | {nlo} | "
                  f"{D.get('peak_MiB', 0):.0f} | {D.get('first_step_s', float('nan')):.3f} |")
