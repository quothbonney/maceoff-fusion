"""gate + timing JSON -> markdown + combined JSON. usage: python3 summ_val7.py OUT.json gate1.json [gate2 ...] -- time1.json [time2 ...]"""
import json, sys
a = sys.argv[2:]; i = a.index("--"); gates, times = a[:i], a[i + 1:]
out = dict(gates={}, timing={})
print("## Correctness gate (graphed7 GraphedRunner + fusion transforms vs eager stock e3nn fp64 on the stock batch)\n")
print("Thresholds |dE|/atom <= 1e-11 eV, max|dF| <= 1e-11 eV/A, |dsigma| <= 1e-11 eV/A^3, edge count == stock. worst over rows: dE/atom, max|dF|, |dsigma|\n")
print("| config (job) | test | rows | ok | worst dE/atom | worst max dF | worst dsigma |")
print("|---|---|---|---|---|---|---|")
for g in gates:
    d = json.load(open(g))
    for b, v in d["backends"].items():
        out["gates"][b] = dict(job=d["job"], failures=[f for f in d["failures"] if f.startswith(b + "/")],
                               tests={t: dict(ok=x.get("ok"), worst=x.get("worst"), n=len(x.get("rows", []))) for t, x in v["tests"].items()})
        for t, x in v["tests"].items():
            w = x.get("worst", {})
            print(f"| {b} ({d['job']}) | {t} | {len(x.get('rows', []))} | {x.get('ok')} | {w.get('dE', float('nan')):.1e} | {w.get('dF', float('nan')):.1e} | {w.get('dS', 0):.1e} |")
print()
for tf in times:
    d = json.load(open(tf))
    out["timing"][tf.split("/")[-1]] = d
    print(f"## Timing job {d['job']} ({tf.split('/')[-1]}); GPU start `{d['gpu_start']}` end `{d.get('gpu_end')}`\n")
    print("build+install s per config: " + ", ".join(f"{c} {v['build_install_s']:.2f}" for c, v in d["configs"].items()) + "\n")
    cfgs = list(d["configs"])
    print("| system (N, E/E_cap) | config | replay ms med [p10-p90] | vs base | full step ms med [p10-p90] (GPU build+copy-in+replay) | vs base | GPU ops in graph | first step s (capture) | peak MiB | fused kernels us (% of replay) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, S in d["systems"].items():
        for c in cfgs:
            D = S.get(c)
            if D is None or "error" in D:
                print(f"| {s} | {c} | ERROR {D and D.get('error', '')[:80]} |"); continue
            base = S.get(c.split("+")[0], {})
            r, st = D["t_replay"], D["t_step"]
            vr = f"{r['median'] / base['t_replay']['median'] - 1:+.1%}" if "+" in c and "t_replay" in base else "-"
            vs = f"{st['median'] / base['t_step']['median'] - 1:+.1%}" if "+" in c and "t_step" in base else "-"
            print(f"| {s} ({S['N']}, {D['E']}/{D['E_cap']}) | {c} | {r['median']:.3f} [{r['p10']:.3f}-{r['p90']:.3f}] | {vr} | {st['median']:.3f} [{st['p10']:.3f}-{st['p90']:.3f}] | {vs} | "
                  f"{D['n_gpu_ops']} | {D['first_step_s']:.3f} ({D['capture_s']:.3f}) | {D['peak_MiB']:.0f} | {D['fused_us']:.0f} ({D['fused_frac_of_replay']:.1%}) |")
    print()
json.dump(out, open(sys.argv[1], "w"), indent=1)
