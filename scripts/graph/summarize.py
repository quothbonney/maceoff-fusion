"""Summarize results/graph/graph_run*.json -> table + end-to-end upper bounds (hybrid denominators from bench4_run1)."""
import json, sys, os
R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
runs = sys.argv[1:] or [f"{R}/results/graph/graph_run1.json", f"{R}/results/graph/graph_run2.json"]
b4 = json.load(open(f"{R}/results/perf2/bench4_run1.json"))
HYB = {s: b4["systems"][s]["stats"]["hybrid"]["ase_total"]["median"] for s in b4["systems"]}
G = [json.load(open(p)) for p in runs]
d0 = G[0]["systems"]; names = list(d0)
ORDER = list(d0[names[0]]["parts"])
f = lambda s: f"{s['median']:.3f} [{s['p10']:.2f}-{s['p90']:.2f}]"
print("part (ms) median [p10-p90], run1 | " + " | ".join(names))
for k in ORDER: print(f"{k:45s} | " + " | ".join(f(d0[n]["parts"][k]) for n in names))
for lab, key in [("SUM of part medians", None), ("instrumented total", "instrumented_total"), ("graph_total plain", "graph_total_plain"), ("proto_build", "proto_build")]:
    print(f"{lab:45s} | " + " | ".join((f"{d0[n]['sum_of_part_medians']:.3f}" if key is None else f(d0[n][key])) for n in names))
out = {}
for n in names:
    s = d0[n]; P = {k: s["parts"][k]["median"] for k in ORDER}
    nl = P["nl_matscipy"] + P["nl_mace_wrapper(cell/dirs/selfedge/shifts)"]
    tens = sum(P[k] for k in ORDER if k.startswith("fromcfg") or k in ("collate_from_data_list", "h2d_to_device", "clone_batch_gpu", "dtype_cast_loop", "to_dict"))
    gt = [g["systems"][n]["graph_total_plain"]["median"] for g in G]; pr = [g["systems"][n]["proto_build"]["median"] for g in G]
    gtm, prm = sum(gt) / len(gt), sum(pr) / len(pr)
    den = HYB.get(n)
    o = dict(n_atoms=s["n_atoms"], n_edges=s["n_edges"], graph_total_runs=gt, proto_runs=pr, nl_ms=nl, nl_frac=nl / gtm,
             per_tensor_overhead_ms=tens, tensor_frac=tens / gtm, hybrid_ase_total_ms=den)
    if den:
        o.update(ceiling_saving_ms=gtm, ceiling_frac=gtm / den, ceiling_speedup=den / (den - gtm),
                 proto_saving_ms=gtm - prm, proto_frac=(gtm - prm) / den, proto_speedup=den / (den - gtm + prm),
                 est_new_call_ms=den - gtm + prm)
    out[n] = o
    print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}))
json.dump(dict(runs=runs, hybrid_denominators_from="results/perf2/bench4_run1.json systems.*.stats.hybrid.ase_total.median", summary=out),
          open(f"{R}/results/graph/summary.json", "w"), indent=1)
