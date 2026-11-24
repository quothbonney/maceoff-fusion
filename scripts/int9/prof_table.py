"""where time moved: proto (node-hybrid P8 B2) vs gpustep (P9) from prof9 JSON. usage: python3 prof_table.py prof_final.json"""
import json, sys
d = json.load(open(sys.argv[1]))
C = ["NL_build", "conv_bwd(node)", "gemm(radialMLP+linears)", "elementwise", "fused_node", "conv_fwd(node)", "cueq_SC",
     "csr_sort/search", "scatter/gather/index", "fused_embed", "scan/compact", "reduce", "memcpy/memset", "other"]
print("| system | path | step ms (synced, no profiler) | GPU kernel sum us | " + " | ".join(C) + " | host sync API calls/step |")
print("|---|---|---|---|" + "---|" * len(C) + "---|")
for s in ["wbox5184", "water5184", "wbox2187", "water2187", "wbox648", "aspirin"]:
    for m, lab in (("proto", "node-hybrid + proto"), ("gpustep", "GPUStep")):
        r = d["runs"].get(f"{m}/{s}")
        if not r or "profile" not in r: continue
        p = r["profile"]; cats = p["cats_us_per_step"]
        sy = {k: v for k, v in p["sync_api_calls_per_step"].items() if k != "cudaDeviceSynchronize"}
        print(f"| {s} | {lab} | {r['stage_ms']['step']['median']:.3f} | {sum(cats.values()):.0f} | " +
              " | ".join(f"{cats.get(c, 0):.0f}" for c in C) + f" | {sy} |")
