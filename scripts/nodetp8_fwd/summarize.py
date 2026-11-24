import json, sys
r = json.load(open(sys.argv[1]))
print("job", r["job"], r["gpu_start"], "csr:", r["csr"])
print("| system | L | N | E | G | rel err | max abs | node eager | node graph | node graph x20 | FlashTP eager | FlashTP graph | FlashTP x20 | e3nn eager | e3nn graph | csr graph | node+csr graph | speedup vs FlashTP (x20) | GB/s uniq | TFLOP/s | HBM floor us | FP64 floor us |")
print("|" + "---|" * 23)
f = lambda d: f"{d['median']*1e3:.1f} [{d['p10']*1e3:.1f}-{d['p90']*1e3:.1f}]"
for s, v in r["systems"].items():
    for li, x in v["layers"].items():
        T, a = x["timing_ms"], x["achieved"]
        print(f"| {s} | {li} | {x['N']} | {x['E']} | {x['G']} | {x['err']['rel_to_max']:.1e} | {x['err']['max_abs']:.1e} | {f(T['node_fwd'])} | {f(T['node_fwd_graph'])} | "
              f"{T['node_fwd_graph_x20']['median']*1e3:.1f} | {f(T['flashtp_fwd'])} | {f(T['flashtp_fwd_graph'])} | {T['flashtp_fwd_graph_x20']['median']*1e3:.1f} | "
              f"{f(T['e3nn_fwd'])} | {f(T['e3nn_fwd_graph'])} | {f(T['csr_build_graph'])} | {f(T['node_csr_fwd_graph'])} | "
              f"{T['flashtp_fwd_graph_x20']['median']/T['node_fwd_graph_x20']['median']:.1f}x | {a['GBps_unique_bytes']:.0f} | {a['TFLOPs']:.2f} | {a['roofline_us_hbm']:.1f} | {a['roofline_us_fp64']:.1f} |")
for k, x in r.get("synthetic", {}).items():
    print("synthetic", k, "N", x["N"], "E", x["E"], "rel", f"{x['err']['rel_to_max']:.1e}", "maxabs", f"{x['err']['max_abs']:.1e}", "near0 n", x["err"]["n_near_zero"], "near0 maxabs", x["err"]["max_abs_near_zero"])
print("near-zero (|ref|<1e-8) per system:", {f"{s}/L{li}": (x["err"]["n_near_zero"], x["err"]["max_abs_near_zero"]) for s, v in r["systems"].items() for li, x in v["layers"].items()})
print("sorted_by_src:", {f"{s}/L{li}": x["edges_sorted_by_src"] for s, v in r["systems"].items() for li, x in v["layers"].items()})
