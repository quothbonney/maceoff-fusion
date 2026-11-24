"""usage: summ8.py gate.json conv.json time.json > tables.md ; also writes results/prof8/INTEGRATION.json"""
import json, sys
G, C, T = [json.load(open(p)) if p != "-" else None for p in sys.argv[1:4]]
out = {}
if G:
    print(f"### Correctness gate (job {G['job']}), tol dE/atom {G['tol']['dE_per_atom']}, dF {G['tol']['max_dF']}, dsigma {G['tol']['dstress']}\n")
    print("| config | test | ok | n | worst dE/atom | worst max dF | worst dsigma |\n|---|---|---|---|---|---|---|")
    for bk, B in G["backends"].items():
        for tn, t in B["tests"].items():
            w = t.get("worst", {})
            print(f"| {bk} | {tn} | {t['ok']} | {len(t.get('rows', []))} | {w.get('dE', float('nan')):.1e} | {w.get('dF', float('nan')):.1e} | {w.get('dS', float('nan')):.1e} |")
    print(f"\nfailures: {G['failures']}\n")
    out["gate"] = dict(job=G["job"], failures=G["failures"])
if C:
    impls = [k for k in ["flash", "node", "e3nn", "oeq", "luna"] if any(k in D for D in C["data"].values())]
    print(f"### Conv-only (job {C['job']}), real captured inputs, ms median [p10-p90]; gpu = sum of kernel time per fwd+bwd (us)\n")
    print("| system/layer | N/E | impl | max rel err (out,gx,gy,gw) | eager fwd | eager bwd | eager fwd+bwd | graph fwd | graph fwd+bwd | graph bwd (diff) | GPU us fwd / fwd+bwd | kernels f+b | peak MiB |\n" + "|---" * 13 + "|")
    f = lambda d: f"{d['median']:.3f} [{d['p10']:.3f}-{d['p90']:.3f}]"
    for key, D in C["data"].items():
        for k in impls:
            if k not in D: continue
            t = D[k]
            if "error" in t:
                print(f"| {key} | {D['N']}/{D['E']} | {k} | ERROR {t['error'][:80]} |"); continue
            rel = max(v["rel"] for v in t["err"].values())
            print(f"| {key} | {D['N']}/{D['E']} | {k} | {rel:.1e} | {f(t['eager_fwd'])} | {f(t['eager_bwd'])} | {f(t['eager_fwdbwd'])} | {f(t['graph_fwd'])} | "
                  f"{f(t['graph_fwdbwd'])} | {t['graph_bwd_est']:.3f} | {t['gpu_us_fwd']:.1f} / {t['gpu_us_fwdbwd']:.1f} | {t['n_gpu_ops_fwdbwd']:.0f} | {t['peak_MiB_fwdbwd']:.0f} |")
        if "csr_build" in D:
            print(f"| {key} | | csr_build only | | {f(D['csr_build'])} | | | | | | | | |")
    # speedup summary per system (sum of both layers, graph fwd+bwd and GPU kernel time)
    print("\n| system | " + " | ".join(f"{k} graph f+b ms (L0+L1)" for k in impls) + " | " + " | ".join(f"{k} GPU us" for k in impls) + " | node vs flash (graph / GPU) |\n" + "|---" * (2 + 2 * len(impls)) + "|")
    syss = sorted({k.split("/")[0] for k in C["data"]}, key=lambda s: C["data"][s + "/L0"]["E"])
    summ = {}
    for s in syss:
        r = {}
        for k in impls:
            try: r[k] = (sum(C["data"][f"{s}/L{i}"][k]["graph_fwdbwd"]["median"] for i in (0, 1)), sum(C["data"][f"{s}/L{i}"][k]["gpu_us_fwdbwd"] for i in (0, 1)))
            except Exception: r[k] = (float("nan"), float("nan"))
        summ[s] = r
        sp = f"{r['flash'][0]/r['node'][0]:.2f}x / {r['flash'][1]/r['node'][1]:.2f}x" if "node" in r and "flash" in r else ""
        print(f"| {s} | " + " | ".join(f"{r[k][0]:.3f}" for k in impls) + " | " + " | ".join(f"{r[k][1]:.0f}" for k in impls) + f" | {sp} |")
    out["conv"] = dict(job=C["job"], summary=summ)
if T:
    cf = list(T["configs"])
    print(f"\n### Whole model, graphed (job {T['job']}); replay / full step ms median [p10-p90]; conv = conv-kernel GPU time in one replay\n")
    print("| system (N,E) | config | replay | full step | first step s | capture s | GPU ops | conv us (frac of replay) | peak MiB |\n|---|---|---|---|---|---|---|---|---|")
    for s, S in T["systems"].items():
        for c in cf:
            d = S.get(c, {})
            if "error" in d or not d:
                print(f"| {s} | {c} | ERROR {d.get('error','')[:80]} |"); continue
            f = lambda x: f"{x['median']:.3f} [{x['p10']:.3f}-{x['p90']:.3f}]"
            print(f"| {s} ({S['N']},{d['E']}) | {c} | {f(d['t_replay'])} | {f(d['t_step'])} | {d['first_step_s']:.2f} | {d['capture_s']:.2f} | {d['n_gpu_ops']} | {d['fused_us']:.0f} ({100*d['fused_frac_of_replay']:.1f}%) | {d['peak_MiB']:.0f} |")
    out["whole"] = dict(job=T["job"], configs=T["configs"], systems={s: {c: {k: S[c][k] for k in ("t_replay", "t_step", "n_gpu_ops", "fused_us", "fused_frac_of_replay", "peak_MiB", "first_step_s") if k in S.get(c, {})} for c in cf} for s, S in T["systems"].items()})
json.dump(out, open("/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/results/prof8/INTEGRATION.json", "w"), indent=1)
