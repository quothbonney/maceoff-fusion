import json
R = "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/results/prof8"
S = json.load(open(f"{R}/flashtp_ncu_summary.json"))["kernels"]
T = json.load(open(f"{R}/flash_kernel_times.json"))["systems"]
CFG = {0: dict(d1=128, dout=2048, W=512, nnz=16, od=16), 1: dict(d1=512, dout=5120, W=1280, nnz=86, od=40)}
BW, FP64 = 3.35e12, 33.5e12  # H100 SXM HBM3 nominal, FP64 non-tensor
rows = []; floors = {}
for k in S:
    s, L, d = k["system"], k["layer"], k["dir"]
    t = T[f"{s}/L{L}"]; N, E = t["N"], t["E"]; c = CFG[L]
    kn = [n for n in t["kernels"] if ("fwd_sptp" if d == "fwd" else "bwd_sptp") in n][0]
    tus = t["kernels"][kn]["median_us"]
    fl = E * 128 * (3 * c["nnz"] + c["od"]) * (1 if d == "fwd" else 2.5)
    if d == "fwd":
        by = 8 * (E * (c["W"] + 16) + N * (c["d1"] + c["dout"])) + 8 * E
    else:
        by = 8 * (2 * E * (c["W"] + 16) + N * (2 * c["d1"] + c["dout"])) + 8 * E
    fdram, ffp = by / BW * 1e6, fl / FP64 * 1e6
    floors[f"{s}/L{L}/{d}"] = dict(N=N, E=E, t_us=tus, bytes=by, flops=fl, floor_dram_us=fdram, floor_fp64_us=ffp)
    g = eval(k["grid"]); ctas = g[0] * g[1]
    rows.append(f"| {s} | L{L} {d} | {N}/{E} | {tus:.1f} | {float(k['dur_us'])*1e3:.1f} | {k['grid']}={ctas} x {k['block']} | {float(k['waves']):.2f} | "
                f"{k['regs']} / {float(k['smem_dyn_B']):.1f} KB | {float(k['occ_theor_%']):.1f} / {float(k['occ_ach_%']):.1f} | "
                f"{float(k['issue_%']):.0f} | {float(k['fp64_pipe_%']):.1f} | {float(k['dram_%']):.1f} | {float(k['l2_%']):.1f} | {float(k['l1_%']):.0f} | "
                f"{float(k['red_global_inst'])/1e3:.0f}k | {', '.join(f'{a} {b:.0f}' for a, b in k['top_stalls_pcsamp'][:3])} | "
                f"{fl/tus/1e6:.2f} | {by/tus/1e3:.0f} | {fdram:.1f} / {ffp:.1f} |")
hdr = ("| system | kernel | N/E | t_us (boost, torch.profiler) | t_us (ncu, 1.59 GHz lock) | grid x block | waves | regs / dyn smem | occ theor/ach % | issue % | "
       "FP64 pipe % | DRAM % | L2 % | L1/shared % | RED insts (warp) | top stalls (pc-sample %) | achieved TFLOP/s | achieved GB/s (compulsory) | floor DRAM / FP64 us |\n" + "|---" * 19 + "|")
print(hdr); print("\n".join(rows))
tot = {}
for key, v in floors.items():
    s = key.split("/")[0]; a = tot.setdefault(s, dict(t=0, fd=0, ff=0)); a["t"] += v["t_us"]; a["fd"] += v["floor_dram_us"]; a["ff"] += v["floor_fp64_us"]
print("\n| system | FlashTP conv fwd+bwd both layers (us) | DRAM-compulsory floor (us) | FP64 floor (us) | gap x |\n|---|---|---|---|---|")
for s, a in tot.items():
    print(f"| {s} | {a['t']:.1f} | {a['fd']:.1f} | {a['ff']:.1f} | {a['t']/max(a['fd'],a['ff']):.1f} |")
json.dump(dict(per_kernel=floors, per_system=tot), open(f"{R}/flashtp_floors.json", "w"), indent=1)
