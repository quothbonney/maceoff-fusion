"""Summarise results/cudagraph6/cg6_<tag>.json -> markdown tables on stdout. usage: python summarize.py cg6_run1.json"""
import json, sys
r = json.load(open(sys.argv[1]))
print(f"job {r['job']} node {r['node']} | start {r['gpu_start']} | end {r.get('gpu_end')}\n")
print("## Correctness (graphed vs eager e3nn on unpadded graph; fixed + 5-step 0.02 A random walk)")
print("| system | N | E (seq) | E_cap | backend | max dE/atom eV | max abs dF eV/A | max RMS dF | graphed vs eager same backend max dF | repeat-replay max dF | finite |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s, S in r["systems"].items():
    for b, B in S["backends"].items():
        if "error" in B:
            print(f"| {s} | {S['n_atoms']} | | | {b} | ERROR {B['error'][:80]} |"); continue
        cs = B["correctness_seq"]
        print(f"| {s} | {S['n_atoms']} | {min(S['seq_edges'])}-{max(S['seq_edges'])} | {S['E_cap']} | {b} | "
              f"{max(c['vs_e3nn']['dE_per_atom'] for c in cs):.1e} | {max(c['vs_e3nn']['max_dF'] for c in cs):.1e} | "
              f"{max(c['vs_e3nn']['rms_dF'] for c in cs):.1e} | {max(c['vs_eager_same']['max_dF'] for c in cs):.1e} | "
              f"{B['repeat_replay']['max_dF']:.1e} | {all(c['vs_e3nn']['finite'] for c in cs)} |")
print("\n## Timing, ms median [p10-p90] (n=%d after %d warmups, cuda-synced)" % (r["reps"], r["warm"]))
print("| system | backend | eager model | graphed copy-in+replay | replay only | speedup model | e2e eager (GPU build) | e2e graphed (GPU build) | speedup e2e | GPU build only | capture s | recapture s | graph pool MiB | launch API calls eager -> graphed |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
f = lambda t: f"{t['median']:.2f} [{t['p10']:.2f}-{t['p90']:.2f}]"
for s, S in r["systems"].items():
    for b, B in S["backends"].items():
        if "error" in B:
            continue
        em, gm = B["t_eager_model"]["median"], B["t_graphed_model_copyin_replay"]["median"]
        ee, ge = B["t_e2e_eager_gpubuild"]["median"], B["t_e2e_graphed_gpubuild"]["median"]
        print(f"| {s} | {b} | {f(B['t_eager_model'])} | {f(B['t_graphed_model_copyin_replay'])} | {f(B['t_graphed_replay_only'])} | "
              f"{em/gm:.1f}x | {f(B['t_e2e_eager_gpubuild'])} | {f(B['t_e2e_graphed_gpubuild'])} | {ee/ge:.1f}x | "
              f"{f(B['t_gpubuild_only'])} | {B['capture_s']:.2f} | {B['recapture_s']:.2f} | {B['peak_mem_capture_MiB']:.0f} | "
              f"{B['api_eager_model']['n_launch_api']} -> {B['api_graphed_step']['n_launch_api']} |")
print("\nfailures:", r["failures"])
