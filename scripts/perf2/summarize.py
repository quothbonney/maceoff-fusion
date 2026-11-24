import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p)); B = d["backends"]
    print(f"\n### {p.split('/')[-1]}  job {d['slurm_job']}  {d['nvidia_smi_start']}  ->end {d.get('nvidia_smi_end')}")
    print("setup ms:", {be: {k: round(v, 1) for k, v in s.items() if k.endswith('_ms')} for be, s in d["setup"].items()})
    for n, s in d["systems"].items():
        print(f"\n{n} atoms={s['n_atoms']} edges={s['n_edges']}")
        print("parity:", {be: (f"{s['parity'][be]['abs_dE_per_atom_eV']:.1e}", f"{s['parity'][be]['max_abs_dF_eVA']:.1e}", s['parity'][be]['valid']) for be in B})
        print("mem peak-baseline MiB:", {be: round(s['memory'][be]['peak_minus_baseline_MiB'], 1) for be in B}, " first call ms:", {be: round(v, 1) for be, v in s['first_call_this_system_ms'].items()})
        print("| boundary | " + " | ".join(B) + " |")
        for bn in s["stats"]["e3nn"]:
            cells = []
            for be in B:
                st = s["stats"][be][bn]; sp = s["speedup_median_vs_e3nn"][be][bn]
                cells.append(f"{st['median']:.2f} [{st['p10']:.2f}-{st['p90']:.2f}]" + ("" if be == "e3nn" else f" {sp:.2f}x"))
            print(f"| {bn} | " + " | ".join(cells) + " |")
