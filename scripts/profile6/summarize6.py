"""raw_<tag>.json -> results/profile6/profile.json + SUMMARY.md tables (SUMMARY prose added by hand below a marker)."""
import json, sys, re, collections, os
R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study"); RES = f"{R}/results/profile6"
raw = json.load(open(f"{RES}/raw_{sys.argv[1]}.json"))
BES = list(raw["backends"]); SYS = list(raw["backends"][BES[0]]["systems"]); PH = ("fwd", "bwd", "fwd_bwd")
L = []; P = L.append


def sync_sites(syncs):
    c = collections.Counter((s["api"], s["annot"] or "-", (s["cpu_op"] or "-")[:60]) for s in syncs if not s["explicit_phase_end"])
    return [dict(api=a, annot=b, cpu_op=o, count=n) for (a, b, o), n in c.most_common()]


def merge_layers(per):
    m = collections.defaultdict(lambda: collections.Counter())
    for k, v in per.items():
        kk = re.sub(r"^[LP]\d+\.", lambda x: x.group(0)[0] + "*.", k)
        if kk.startswith("bwd_unmapped"): kk = "bwd_unmapped"
        for f in ("gpu_us", "cpu_us", "launches", "bwd_gpu_us", "bwd_cpu_us", "bwd_launches"): m[kk][f] += v.get(f, 0)
    return m


prof = dict(meta={k: raw[k] for k in ("slurm_job", "node", "nvidia_smi_start", "nvidia_smi_end", "torch", "nprof", "ntime", "warm", "check_indices")},
            graph_construction=raw["graph_construction"], results={})
for be in BES:
    for s in SYS:
        d = raw["backends"][be]["systems"][s]
        for ph in PH:
            p = d["profiled"][ph]; t = d["timing_unprofiled"][ph]
            r = dict(backend=be, system=s, phase=ph, n_atoms=d["n_atoms"], n_edges=d["n_edges"],
                     wall_ms_unprofiled=t["wall_ms_median"], cuda_event_ms_unprofiled=t["cuda_event_ms_median"],
                     wall_ms_profiled=p["wall_us"] / 1e3, kernel_sum_ms=p["kernel_sum_us"] / 1e3, busy_union_ms=p["busy_union_us"] / 1e3,
                     idle_frac_profiled=p["idle_frac"], idle_frac_vs_unprofiled_wall=max(0.0, 1 - p["busy_union_us"] / 1e3 / t["wall_ms_median"]),
                     n_kernels=p["n_kernels"], n_launch_api=p["n_launch_api"], launch_api_ms=p["launch_api_us"] / 1e3,
                     n_memcpy=p["n_memcpy"], n_memset=p["n_memset"], memcpy_kinds=p["memcpy_kinds"], runtime_api_counts=p["runtime_api_counts"],
                     sync_points=sync_sites(p["syncs"]), n_sync_points=sum(1 for x in p["syncs"] if not x["explicit_phase_end"]),
                     sync_ms=sum(x["dur_us"] for x in p["syncs"] if not x["explicit_phase_end"]) / 1e3,
                     per_op=p["per_op"], top_kernels=[dict(name=n, us=u, count=c) for n, u, c in p["top_kernels"][:10]])
            prof["results"][f"{be}|{s}|{ph}"] = r
        for k in ("intermediate_shapes", "saved_for_backward", "saved_for_backward_total_MB"):
            if k in d: prof["results"][f"{be}|{s}|fwd"][k] = d[k]
json.dump(prof, open(f"{RES}/profile.json", "w"), indent=1)

P(f"# MACE-OFF23 medium: model-compute profile (profile6)\n\nSLURM job {raw['slurm_job']} on {raw['node']}; GPU: `{raw['nvidia_smi_start']}` (end: `{raw['nvidia_smi_end']}`). "
  f"torch {raw['torch']}, fp64, FlashTP check_indices=False. Graph built once per system (stock make_batch_dict), kept on GPU. "
  f"Unprofiled timings: median of {raw['ntime']} synced reps (no hooks). Profiled: torch.profiler (CPU+CUDA, record_shapes, no stack), "
  f"{raw['nprof']} iterations/phase with module annotations; per-iteration median. Traces: `results/profile6/trace_<backend>_<system>.json.gz`.\n")
P("## 1. Phase totals\nwall = unprofiled synced wall (ms); ev = CUDA-event ms; pwall = profiled wall; ksum = sum of kernel durations; busy = union of kernel+memcpy+memset intervals; "
  "idle = 1 - busy/wall (unprofiled wall); nK = kernels; launchAPI = CPU time in cudaLaunchKernel/cuLaunchKernel/ExC (profiled); mc/ms = memcpy/memset count; sync = implicit host syncs (count, ms).\n")
P("| backend | system | phase | wall | ev | pwall | ksum | busy | idle | nK | launchAPI ms | mc/ms | sync (n, ms) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in SYS:
    for be in BES:
        for ph in PH:
            r = prof["results"][f"{be}|{s}|{ph}"]
            P(f"| {be} | {s} ({r['n_atoms']}a/{r['n_edges']}e) | {ph} | {r['wall_ms_unprofiled']:.2f} | {r['cuda_event_ms_unprofiled']:.2f} | {r['wall_ms_profiled']:.2f} | "
              f"{r['kernel_sum_ms']:.2f} | {r['busy_union_ms']:.2f} | {r['idle_frac_vs_unprofiled_wall']:.2f} | {r['n_kernels']:.0f} | {r['launch_api_ms']:.2f} | "
              f"{r['n_memcpy']:.0f}/{r['n_memset']:.0f} | {r['n_sync_points']}, {r['sync_ms']:.2f} |")
P("\n### Host-sync call sites (per call, fwd and bwd phases; explicit end-of-phase torch.cuda.synchronize excluded)\n")
seen = set()
for be in BES:
    for ph in ("fwd", "bwd"):
        r = prof["results"][f"{be}|water648|{ph}"]
        P(f"- **{be} {ph}**: " + ("; ".join(f"{x['count']}x {x['api']} in `{x['annot']}` / `{x['cpu_op']}`" for x in r["sync_points"]) or "none"))
for gi in ("aspirin", "water648"):
    P(f"\n## 2. Time by MACE operation: {gi} (layers merged L*/P*; per-layer in profile.json)\nGPU = sum of kernel time (us) launched inside the op's annotation (fwd) or by autograd nodes whose sequence number maps to a forward op inside the annotation (bwd); "
      "CPU = exclusive annotation time (fwd) / evaluate_function time (bwd), profiled (inflated ~1.0-1.6x by the profiler). Phase = fwd for fwd columns, bwd phase for bwd columns.\n")
    for be in BES:
        f = merge_layers(prof["results"][f"{be}|{gi}|fwd"]["per_op"]); b = merge_layers(prof["results"][f"{be}|{gi}|bwd"]["per_op"])
        keys = sorted(set(f) | set(b), key=lambda k: -(f[k]["gpu_us"] + b[k]["bwd_gpu_us"] + 1e-3 * (f[k]["cpu_us"] + b[k]["bwd_cpu_us"])))
        P(f"\n**{be}**\n\n| op | fwd GPU us | fwd CPU us | fwd launches | bwd GPU us | bwd CPU us | bwd launches |\n|---|---|---|---|---|---|---|")
        for k in keys:
            if f[k]["gpu_us"] + f[k]["cpu_us"] + b[k]["bwd_gpu_us"] + b[k]["bwd_cpu_us"] < 1: continue
            P(f"| {k} | {f[k]['gpu_us']:.0f} | {f[k]['cpu_us']:.0f} | {f[k]['launches']} | {b[k]['bwd_gpu_us']:.0f} | {b[k]['bwd_cpu_us']:.0f} | {b[k]['bwd_launches']} |")
P("\n## 3. Intermediate tensors and saved-for-backward\n")
for gi in ("aspirin", "water648"):
    sh = prof["results"][f"e3nn|{gi}|fwd"].get("intermediate_shapes", {})
    P(f"\n**{gi}: module outputs (e3nn backend; flash/hybrid identical shapes except where noted in profile.json)**\n\n| module | outputs (shape dtype MB) |\n|---|---|")
    for k, v in sh.items():
        if k == "model": continue
        P(f"| {k} | " + "; ".join(f"{x['shape']} {x['dtype']} {x['MB']:.2f}" for x in v) + " |")
    P(f"\n**{gi}: tensors saved for backward (unique CUDA storages, MB; count = pack calls)**\n\n| op | " + " | ".join(BES) + " |\n|---|" + "---|" * len(BES))
    sv = {be: merge_layers({k: dict(gpu_us=v["MB"], launches=v["count"]) for k, v in prof["results"][f"{be}|{gi}|fwd"].get("saved_for_backward", {}).items()}) for be in BES}
    for k in sorted(set().union(*[set(x) for x in sv.values()]), key=lambda k: -max(sv[b][k]["gpu_us"] for b in BES)):
        P(f"| {k} | " + " | ".join(f"{sv[b][k]['gpu_us']:.2f} ({sv[b][k]['launches']})" for b in BES) + " |")
    P("| **total MB** | " + " | ".join(f"{prof['results'][f'{b}|{gi}|fwd'].get('saved_for_backward_total_MB', 0):.1f}" for b in BES) + " |")
P("\n## 4. Stock graph construction (make_batch_dict; measured once, backend-independent)\n\n| system | graph_cpu ms (config+AtomicData) | graph_total ms (incl. collate+H2D+fp64) | total-cpu ms | H2D/D2D memcpys | cudaStreamSynchronize | kernels |\n|---|---|---|---|---|---|---|")
mk = lambda g: ', '.join(k.split(' (')[0] + ':' + str(v) for k, v in g['memcpy_kinds'].items())
for s, g in raw["graph_construction"].items():
    P(f"| {s} | {g['graph_cpu_ms_median']:.2f} | {g['graph_total_ms_median']:.2f} | {g['h2d_tensor_part_ms']:.2f} | {g['n_memcpy']} ({mk(g)}) | {g['runtime_api_counts'].get('cudaStreamSynchronize', 0)} | {g['n_kernels']} |")
P("\n## 5. Top-10 kernels (fwd_bwd phase = model(compute_force=True), one iteration)\n")
for gi in ("aspirin", "water648"):
    for be in BES:
        r = prof["results"][f"{be}|{gi}|fwd_bwd"]; tot = r["kernel_sum_ms"] * 1e3
        P(f"\n**{be} / {gi}** (kernel sum {tot / 1e3:.2f} ms)\n\n| kernel | us | % | n |\n|---|---|---|---|")
        for k in r["top_kernels"]:
            P(f"| `{k['name'][:110]}` | {k['us']:.0f} | {100 * k['us'] / tot:.1f} | {k['count']} |")
open(f"{RES}/SUMMARY_tables.md", "w").write("\n".join(L) + "\n")
print("ok")
