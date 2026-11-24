"""Summarize results/nl9/tests.json (+ timing.json if present) as text tables. usage: python -I summ_nl.py"""
import json, os, re, collections
R = os.environ["R"]
T = json.load(open(f"{R}/results/nl9/tests.json"))
groups = collections.OrderedDict()
for t in T["tests"]:
    name = t["test"]
    g = re.sub(r"/(walk|replay|cfg|perturb|seed)\d+$", "", name)
    g = re.sub(r"/seed\d+", "", g)
    groups.setdefault(g, []).append(t)
print(f"tests: job {T['job']} {T['gpu'].split(',')[2].strip() if ',' in T['gpu'] else ''}  {T['n_tests']} tests, "
      f"{T['n_fail']} failures")
for g, ts in groups.items():
    ok = sum(t["ok"] for t in ts)
    extra = ""
    e = [t.get("E") for t in ts if t.get("E") is not None]
    if e and isinstance(e[0], int):
        extra += f" E={min(e)}..{max(e)}" if min(e) != max(e) else f" E={e[0]}"
    if g.startswith("edges/") or g.startswith("boundary") or g.startswith("cudagraph"):
        ks = ["eq_proto", "eq_stock_set", "eq_stock_shifts"]
        vals = {k: all(t.get(k, True) for t in ts) for k in ks if any(k in t for t in ts)}
        extra += " " + " ".join(f"{k}={'Y' if v else 'n'}" for k, v in vals.items())
        sd = [t.get("stock_symdiff", 0) for t in ts]
        if any(sd):
            extra += f" stock_symdiff={sum(sd)} (max|d-r|={max(t.get('stock_symdiff_max_abs_d_minus_rmax', 0) for t in ts):.1e})"
        csr = all(all(t.get(k, True) for k in ("rowptr_monotone", "dst_sorted", "row_src_strictly_increasing",
                                                 "perm_src_src_sorted", "reverse_edge_map", "padding", "eq_build_csr"))
                  for t in ts)
        if any("eq_build_csr" in t for t in ts):
            extra += f" csr={'Y' if csr else 'n'}"
    if g.startswith("model/"):
        def mx(key, f):
            return max(t[key][f] for t in ts if key in t)
        try:
            extra += (" padded-vs-stock dE/at {:.1e} dF {:.1e} dS {:.1e}; trim-vs-proto dE/at {:.1e} dF {:.1e} dS {:.1e};"
                      " floor dF {:.1e} dS {:.1e}").format(
                mx("padded_vs_stock", "dE_per_atom"), mx("padded_vs_stock", "max_dF"), mx("padded_vs_stock", "dstress"),
                mx("trim_vs_proto", "dE_per_atom"), mx("trim_vs_proto", "max_dF"), mx("trim_vs_proto", "dstress"),
                mx("floor_proto_stock_and_repeat", "max_dF"), mx("floor_proto_stock_and_repeat", "dstress"))
            strict = all(t[k]["strict_tol"] for t in ts for k in t if isinstance(t.get(k), dict) and "strict_tol" in t[k])
            extra += f" strict_tol={'Y' if strict else 'n (floor-limited)'}"
        except Exception:
            pass
    for k in ("flags", "max_bin", "mismatches", "msg"):
        if k in ts[0]:
            extra += f" {k}={ts[0][k]}"
    print(f"  {'PASS' if ok == len(ts) else 'FAIL'} {g:42s} {ok}/{len(ts)}{extra}")
p = f"{R}/results/nl9/timing.json"
if os.path.exists(p):
    Tm = json.load(open(p))
    print(f"\ntiming: job {Tm['job']} {Tm['gpu_start']}  warmup {Tm['warmup']} reps {Tm['reps']} (proto/matscipy 30 reps for N>3000)")
    print("  squeue at start: " + Tm["squeue_at_start"].replace("\n", "; "))
    print(f"  {'system':14s} {'N':>5s} {'E':>7s}  {'cell eager ms':>22s} {'cell graph ms':>22s} {'graph dev':>9s} "
          f"{'proto_graph ms':>24s} {'matscipy CPU ms':>24s}  nl MiB  proto MiB")
    f = lambda d: f"{d['median_ms']:.3f} [{d['p10_ms']:.3f}-{d['p90_ms']:.3f}]"
    for r in Tm["rows"]:
        print(f"  {r['system']:14s} {r['N']:5d} {r['E']:7d}  {f(r['cell_eager']):>22s} {f(r['cell_graph']):>22s} "
              f"{r['cell_graph_device']['median_ms']:9.3f} {f(r['proto_build']):>24s} {f(r['stock_get_neighborhood_cpu']):>24s}"
              f"  {r['nl_buffers_bytes']/2**20:6.2f}  {r['proto_peak_alloc_bytes']/2**20:8.1f}")
