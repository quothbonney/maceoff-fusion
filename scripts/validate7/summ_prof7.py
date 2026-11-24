"""prof7 JSON -> markdown tables (printed). usage: python3 summ_prof7.py prof7_run1.json"""
import json, sys, math, re, collections

d = json.load(open(sys.argv[1]))
HBM, F64T, F64 = 3.35e12, 67e12, 34e12

TARGET = collections.OrderedDict([
    ("FlashTP conv (fwd+bwd kernels, gather+TP+scatter)", ["conv(FlashTP)", "conv.glue_after(scatter/reshape)"]),
    ("SC (cueq / e3nn)", ["SC(cueq/e3nn)"]),
    ("embed: SH  [B]", ["embed.SH", "embed.glue(after SH)"]),
    ("embed: radial basis+cutoff  [B]", ["embed.radial_basis+cutoff", "embed.glue(after radial)"]),
    ("embed: edge vectors/lengths + e0 (prepare_graph)  [B optional]", ["prepare_graph/e0", "embed.edge_vectors/lengths(prepare_graph)"]),
    ("embed: radial MLP (conv_tp_weights)", ["embed.radial_MLP"]),
    ("interaction linear (+1/avg)  [C]", ["node.interaction_linear(+avg,reshape)"]),
    ("skip_tp  [C]", ["node.skip_tp"]),
    ("linear_up", ["node.linear_up", "node.interaction_other"]),
    ("products linear (+sc add)", ["products.linear(+sc_add)", "products.other"]),
    ("readouts + energy sum/scale_shift/outputs", ["readouts", "outputs/energy_sum/scale_shift", "scale_shift", "node_embedding"]),
])
EDGE_GROUPS = ["FlashTP conv (fwd+bwd kernels, gather+TP+scatter)", "embed: SH  [B]", "embed: radial basis+cutoff  [B]",
               "embed: edge vectors/lengths + e0 (prepare_graph)  [B optional]", "embed: radial MLP (conv_tp_weights)"]
IO = {  # group -> module labels whose inputs+outputs define min traffic
    "FlashTP conv (fwd+bwd kernels, gather+TP+scatter)": ["interactions.0.conv_tp", "interactions.1.conv_tp"],
    "SC (cueq / e3nn)": ["products.0.symmetric_contractions", "products.1.symmetric_contractions"],
    "embed: SH  [B]": ["spherical_harmonics"], "embed: radial basis+cutoff  [B]": ["radial_embedding"],
    "embed: radial MLP (conv_tp_weights)": ["interactions.0.conv_tp_weights", "interactions.1.conv_tp_weights"],
    "interaction linear (+1/avg)  [C]": ["interactions.0.linear", "interactions.1.linear"],
    "skip_tp  [C]": ["interactions.0.skip_tp", "interactions.1.skip_tp"],
    "linear_up": ["interactions.0.linear_up", "interactions.1.linear_up"],
    "products linear (+sc add)": ["products.0.linear", "products.1.linear"],
}
FL = {"embed: radial MLP (conv_tp_weights)": ["L0.conv_tp_weights(radial MLP)", "L1.conv_tp_weights(radial MLP)"],
      "interaction linear (+1/avg)  [C]": ["L0.linear", "L1.linear"], "linear_up": ["L0.linear_up", "L1.linear_up"],
      "products linear (+sc add)": ["P0.linear", "P1.linear"]}


def kclass(n):
    l = n.lower()
    if "sptp" in l: return "FlashTP"
    if "segmented_polynomial" in l: return "cueq"
    if "gemm" in l or "cutlass" in l or "splitk" in l: return "GEMM(fp64)"
    if l.startswith("memcpy") or "copy" in l or l.startswith("memset") or "fill" in l: return "copy/cat/memset"
    if "index" in l or "scatter" in l or "sort" in l or "radix" in l: return "index/scatter/sort"
    if "reduce" in l: return "reduce"
    if "elementwise" in l or "fused_" in l: return "elementwise"
    return "other"


sysorder = [s for s in ["aspirin", "water192", "org192", "water648", "org648", "wbox648"] if s in d["systems"]]
print(f"job {d['job']} node {d['node']} torch {d['torch']}\n")
print("## 1. Replay wall vs GPU busy (one graph per system; replay only, 60 synced reps after 10 warmups)\n")
print("| system | backend | N/E/E_cap | replay ms median [p10-p90] | copy-in+replay ms | GPU kernel sum ms (profiled replay) | busy/replay wall | span ms | kernels+copies in graph | mean kernel us | tight-cap replay ms (pad cost) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in sysorder:
    for b, D in d["systems"][s].items():
        if "error" in D: print(f"| {s} | {b} | ERROR {D['error'][:80]} |"); continue
        t = D["t_replay"]; ks = D["replay_kernel_sum_us"] / 1e3
        print(f"| {s} | {b} | {D['N']}/{D['E']}/{D['E_cap']} | {t['median']:.3f} [{t['p10']:.3f}-{t['p90']:.3f}] | {D['t_copyin_replay']['median']:.3f} | {ks:.3f} | "
              f"{ks / t['median']:.0%} | {D['replay_span_us'] / 1e3:.3f} | {D['replay_n_gpu_ops']} | {D['replay_kernel_sum_us'] / D['replay_n_gpu_ops']:.1f} | "
              f"{D['t_replay_tight_cap']['median']:.3f} ({(t['median'] - D['t_replay_tight_cap']['median']) / t['median']:+.1%}) |")

print("\n## 2. GPU time by MACE op (fraction of graphed replay wall; fwd/bwd us; #kernels fwd+bwd)\n")
for b in ["hybrid", "flash"]:
    rows = [s for s in sysorder if b in d["systems"][s] and "error" not in d["systems"][s][b]]
    if not rows: continue
    print(f"### {b}\n")
    print("| op group | " + " | ".join(f"{s} (E_cap {d['systems'][s][b]['E_cap']})" for s in rows) + " |")
    print("|---|" + "---|" * len(rows))
    allg = set()
    for s in rows: allg |= set(d["systems"][s][b]["per_op"])
    used = set()
    lines = []
    for name, gs in list(TARGET.items()) + [("unmapped/other", None)]:
        if gs is None: gs = sorted(allg - used)
        used |= set(gs)
        cells = []; tot0 = 0
        for s in rows:
            D = d["systems"][s][b]; po = D["per_op"]
            f = sum(po[g]["fwd_us"] for g in gs if g in po); bw = sum(po[g]["bwd_us"] for g in gs if g in po)
            n = sum(po[g]["n_fwd"] + po[g]["n_bwd"] for g in gs if g in po)
            w = D["t_replay"]["median"] * 1e3
            cells.append(f"{(f + bw) / w:.1%} ({f:.0f}/{bw:.0f}; {n})"); tot0 += (f + bw) / w
        lines.append((tot0, f"| {name} | " + " | ".join(cells) + " |"))
    for _, l in lines: print(l)
    cells = []
    for s in rows:
        D = d["systems"][s][b]; w = D["t_replay"]["median"] * 1e3
        cells.append(f"{1 - D['replay_busy_us'] / w:.1%}")
    print("| GPU idle (launch gaps inside graph) | " + " | ".join(cells) + " |")
    cells = []
    for s in rows:
        D = d["systems"][s][b]; po = D["per_op"]; w = D["t_replay"]["median"] * 1e3
        edge = sum(po[g]["fwd_us"] + po[g]["bwd_us"] for gn in EDGE_GROUPS for g in TARGET[gn] if g in po)
        cells.append(f"~{edge * (1 - D['E'] / D['E_cap']) / w:.1%} est; measured {(D['t_replay']['median'] - D['t_replay_tight_cap']['median']) / D['t_replay']['median']:+.1%}")
    print("| padding overhead (edge-op time x (1-E/E_cap); measured = replay vs tight cap) | " + " | ".join(cells) + " |")
    cells = []
    for s in rows:
        D = d["systems"][s][b]; po = D["per_op"]; w = D["t_replay"]["median"] * 1e3
        cells.append(f"B={sum(po[g]['fwd_us'] + po[g]['bwd_us'] for gn in list(TARGET)[2:4] for g in TARGET[gn] if g in po) / w:.1%} "
                     f"(+edge vec {sum(po[g]['fwd_us'] + po[g]['bwd_us'] for g in TARGET[list(TARGET)[4]] if g in po) / w:.1%}); "
                     f"C={sum(po[g]['fwd_us'] + po[g]['bwd_us'] for gn in list(TARGET)[6:8] for g in TARGET[gn] if g in po) / w:.1%}")
    print("| **fusion targets** | " + " | ".join(cells) + " |\n")

print("## 3. GPU time by kernel class (fraction of replay kernel sum)\n")
for b in ["hybrid", "flash"]:
    rows = [s for s in sysorder if b in d["systems"][s] and "error" not in d["systems"][s][b]]
    cls = collections.OrderedDict()
    for s in rows:
        D = d["systems"][s][b]; c = collections.Counter()
        for g, v in D["per_op"].items():
            for n, t in v["top_kernels"]: pass
        for n, t in D["top_kernels"]: c[kclass(n)] += t
        cls[s] = (c, D["replay_kernel_sum_us"])
    keys = ["FlashTP", "cueq", "GEMM(fp64)", "elementwise", "copy/cat/memset", "index/scatter/sort", "reduce", "other"]
    print(f"### {b} (top-25 kernel names classified; remainder = 'not in top 25')\n")
    print("| class | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
    for k in keys + ["not in top 25"]:
        cells = []
        for s in rows:
            c, tot = cls[s]
            v = (tot - sum(c.values())) if k == "not in top 25" else c.get(k, 0)
            cells.append(f"{v / tot:.1%} ({v:.0f} us)")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print()

print("## 4. Roofline-style estimates (min bytes = module inputs+outputs fwd, x3 for fwd+bwd; FLOPs = 2*MACs fwd, x3 fwd+bwd)\n")
for b in ["hybrid"]:
    for s in [x for x in sysorder if x in ("water648", "wbox648", "water192", "aspirin")]:
        D = d["systems"][s].get(b)
        if not D or "error" in D: continue
        print(f"### {b} {s}\n")
        print("| op group | GPU us (fwd+bwd) | min MB moved | achieved GB/s (% of 3.35 TB/s) | GFLOP | achieved TFLOP/s (% of 67 tensor) |")
        print("|---|---|---|---|---|---|")
        po = D["per_op"]; io = D["io_bytes"]
        for name, labs in IO.items():
            t = sum(po[g]["fwd_us"] + po[g]["bwd_us"] for g in TARGET[name] if g in po)
            mb = 3 * sum(io[l]["in_MB"] + io[l]["out_MB"] for l in labs if l in io)
            fl = sum(D["flops"].get(k) or 0 for k in FL.get(name, [])) * 3
            bw = mb * 1e6 / (t * 1e-6) if t else 0
            ft = f"{fl / 1e9:.2f} | {fl / (t * 1e-6) / 1e12:.2f} ({fl / (t * 1e-6) / F64T:.1%})" if fl else "- | -"
            print(f"| {name} | {t:.0f} | {mb:.1f} | {bw / 1e9:.0f} ({bw / HBM:.1%}) | {ft} |")
        print()
