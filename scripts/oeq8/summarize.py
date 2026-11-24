"""Merge bench_ref/bench_oeq/bench_luna JSON into results/oeq8/baselines.json and print markdown tables."""
import json, os, sys
R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
D = f"{R}/results/oeq8"
srcs = {"ref": "bench_ref.json", "oeq": "bench_oeq.json", "luna": "bench_luna.json"}
merged = {"sources": {}, "rows": {}}
for k, f in srcs.items():
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    d = json.load(open(p)); merged["sources"][k] = dict(file=f, env=d.get("env"), E0=d.get("E0"), build=d.get("build"))
    for key, r in d["results"].items():
        row = merged["rows"].setdefault(key, dict(N=r["N"], E=r["E"]))
        for bn, v in r.items():
            if isinstance(v, dict): row[bn] = v
order = ["e3nn", "flashtp_gc", "cueq_fused_irmul_T", "oeq_atomic", "oeq_det_full", "oeq_det_kernel", "luna_full", "luna_kernel"]
json.dump(merged, open(f"{D}/baselines.json", "w"), indent=1)


def t(v, k):
    x = v.get("time", {}).get(k, {})
    if "median" in x: return f"{x['median']:.3f} [{x['p10']:.3f}-{x['p90']:.3f}]"
    return "n/a" if "error" in x else "-"


lines = []
for key, row in merged["rows"].items():
    lines.append(f"\n#### {key} (N={row['N']}, E={row['E']})\n")
    lines.append("| backend | max scaled err (out/gx/gy/gw) | max abs | det | eager fwd | eager bwd | eager fwd+bwd | graph fwd | graph bwd | graph fwd+bwd |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for bn in order:
        v = row.get(bn)
        if not v: continue
        if "err" not in v:
            lines.append(f"| {bn} | ERROR | | | | | | | | |"); continue
        e = v["err"]
        sc = "/".join(f"{e[k]['scaled']:.1e}" for k in ("out", "gx", "gy", "gw"))
        ma = max(e[k]["max_abs"] for k in e)
        lines.append(f"| {bn} | {sc} | {ma:.1e} | {'Y' if v.get('run_to_run_bitwise') else 'N'} | "
                     + " | ".join(t(v, f"{m}_{s}") for m in ("eager", "graph") for s in ("fwd", "bwd", "fwd_bwd")) + " |")
open(f"{D}/tables.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
