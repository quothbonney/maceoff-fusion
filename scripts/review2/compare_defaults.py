import sys, json, numpy as np
a, b, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
res = {"only_in_f32": sorted(set(a) - set(b)), "only_in_f64": sorted(set(b) - set(a)), "keys": {}}
for k in sorted(set(a) & set(b)):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        res["keys"][k] = f"shape {x.shape} vs {y.shape}"; continue
    res["keys"][k] = dict(bitwise=bool(np.array_equal(x, y)), maxabs=float(np.abs(x.astype(float) - y.astype(float)).max()) if x.size else 0.0)
res["cg_and_meta_all_bitwise"] = all(v["bitwise"] for k, v in res["keys"].items() if isinstance(v, dict) and (k.startswith("cg_") or k.startswith("meta_")))
res["max_output_diff"] = max((v["maxabs"] for k, v in res["keys"].items() if isinstance(v, dict) and not (k.startswith("cg_") or k.startswith("meta_"))), default=None)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("cg_and_meta_all_bitwise", "max_output_diff", "only_in_f32", "only_in_f64")}))
