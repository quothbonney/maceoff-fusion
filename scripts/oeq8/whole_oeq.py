"""venv-oeq: whole-model MACECalculator(enable_oeq=True, fp64) vs stock e3nn on aspirin, water648: E/F diffs and
full calculate() wall time (energy+forces). Usage: python -I whole_oeq.py <out_json>"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import common as C

out = dict(env=C.env_info(), results={})
calcs = dict(e3nn=C.load_calc(), oeq=C.load_calc(enable_oeq=True))
out["oeq_conv_types"] = [type(b.conv_tp).__name__ for b in calcs["oeq"].models[0].interactions]
for name in ["aspirin", "water648"]:
    atoms = C.load_atoms(name); n = len(atoms); r = out["results"][name] = dict(n_atoms=n)
    ef = {}
    for k, calc in calcs.items():
        calc.calculate(atoms, ["energy", "forces"])
        ef[k] = (calc.results["energy"], calc.results["forces"].copy())
        ts = []
        for it in range(60):
            torch.cuda.synchronize(); t0 = time.perf_counter(); calc.calculate(atoms, ["energy", "forces"])
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        r[f"{k}_call_ms"] = C.stats(ts[10:])
    r["dE_per_atom"] = abs(ef["oeq"][0] - ef["e3nn"][0]) / n
    r["max_dF"] = float(np.abs(ef["oeq"][1] - ef["e3nn"][1]).max())
    calcs["oeq"].calculate(atoms, ["energy", "forces"]); e2 = calcs["oeq"].results["energy"]
    r["oeq_rerun_bitwise_E"] = bool(e2 == ef["oeq"][0])
    print(name, json.dumps(r), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
