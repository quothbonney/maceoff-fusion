"""Usage: python -I summarize_wl.py run1.json [run2.json ...]  -> markdown tables to stdout."""
import sys, json
runs = [json.load(open(p)) for p in sys.argv[1:]]
r0 = runs[0]
def f(x, n=2): return "-" if x is None else f"{x:.{n}f}"
print("## Runs"); [print(f"- {p}: job {r['slurm_job']} node {r['node']} backend_set {r['backend_set']} | start {r['nvidia_smi_start']} | end {r.get('nvidia_smi_end')}") for p, r in zip(sys.argv[1:], runs)]
print("\n## Systems\n| system | formula | n_atoms | n_edges | pbc |\n|---|---|---|---|---|")
for n, d in r0["systems"].items(): print(f"| {n} | {d['formula']} | {d['n_atoms']} | {d['n_edges']} | {d['pbc']} |")
print("\n## Correctness (worst over fixed + 5 noisy steps; vs e3nn same process) [run 1]\n| system | backend | max dE/atom eV | max abs dF eV/A | max RMS dF | max abs dsigma eV/A3 | repeat E range eV | repeat max dF | valid |\n|---|---|---|---|---|---|---|---|---|")
for n, d in r0["systems"].items():
    for be, c in d["correctness"]["backends"].items():
        if be == "e3nn": 
            w = c["worst"]; print(f"| {n} | e3nn(ref) | - | - | - | - | {c['repeat']['E_range_eV']:.1e} | {c['repeat']['max_abs_dF_vs_call0']:.1e} | ref |"); continue
        w = c["worst"]
        print(f"| {n} | {be} | {w['dE_per_atom_eV']:.1e} | {w['max_abs_dF_eVA']:.1e} | {w['rms_dF_eVA']:.1e} | {('%.1e' % w['max_abs_dstress_eVA3']) if w.get('max_abs_dstress_eVA3') is not None else ('n/a' if not d['pbc'] else 'FAIL')} | {c['repeat']['E_range_eV']:.1e} | {c['repeat']['max_abs_dF_vs_call0']:.1e} | {w['valid']} |")
B = ["ase_total", "fwd", "bwd", "fwd_bwd", "graph_total"]
for i, r in enumerate(runs):
    if "stats" not in next(iter(r["systems"].values())): continue
    print(f"\n## Timing run {i+1} (job {r['slurm_job']}): median [p10-p90] ms, n per cell")
    print("| system | backend | ase_total | fwd | bwd | fwd_bwd | graph_total | peak-base MiB | first call ms | n |\n|---|---|---|---|---|---|---|---|---|---|")
    for n, d in r["systems"].items():
        for be, st in d["stats"].items():
            cells = [f"{st[b]['median']:.2f} [{st[b]['p10']:.2f}-{st[b]['p90']:.2f}]" for b in B]
            print(f"| {n} | {be} | " + " | ".join(cells) + f" | {d['memory'][be]['peak_minus_baseline_MiB']:.0f} | {d['first_call_this_system_ms'][be]:.0f} | {st['ase_total']['n']} |")
print("\n## Setup (construct + first ASE call on first system), ms")
for i, r in enumerate(runs):
    print(f"run {i+1}: " + "; ".join(f"{be}: construct {s.get('construct_ms',0):.0f}, first {[v for k,v in s.items() if k.startswith('first_')][0]:.0f}, second {[v for k,v in s.items() if k.startswith('second_')][0]:.0f}, total {s.get('setup_total_ms',0):.0f}" for be, s in r["setup"].items() if "construct_ms" in s))
print("\n## Crossover (ase_total median; per run)\n| system | n_atoms | n_edges | " + " | ".join(f"run{i+1} fastest (vs 2nd; vs e3nn)" for i in range(len(runs))) + " |\n|---|---|---|" + "---|" * len(runs))
for n, d in r0["systems"].items():
    cells = []
    for r in runs:
        st = r["systems"].get(n, {}).get("stats")
        if not st: cells.append("-"); continue
        med = sorted((s["ase_total"]["median"], be) for be, s in st.items())
        cells.append(f"{med[0][1]} {med[0][0]:.2f} ms ({med[1][1]} {med[1][0]:.2f}: x{med[1][0]/med[0][0]:.3f}; e3nn x{st['e3nn']['ase_total']['median']/med[0][0]:.2f}); " + ", ".join(f"{be} {m:.2f}" for m, be in med))
    print(f"| {n} | {d['n_atoms']} | {d['n_edges']} | " + " | ".join(cells) + " |")
