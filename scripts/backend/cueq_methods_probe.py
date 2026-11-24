"""List cuequivariance SegmentedPolynomial implementation per module after MACE enable_cueq conversion.
Usage: python -I cueq_methods_probe.py <model_path> <dtype> <out_json>"""
import json, sys, collections
import torch, mace
from mace.calculators import MACECalculator
import cuequivariance_torch as cuet
mp, dt, out = sys.argv[1:4]
calc = MACECalculator(model_paths=mp, device="cuda", default_dtype=dt, compute_stress=False, enable_cueq=True)
rows = []
for name, m in calc.models[0].named_modules():
    if isinstance(m, cuet.SegmentedPolynomial):
        rows.append({"module": name, "method": m.method, "impl": type(m.m).__name__})
cnt = collections.Counter((r["method"], r["impl"]) for r in rows)
for k, v in cnt.items(): print(v, k)
for r in rows: print(r)
json.dump({"dtype": dt, "rows": rows, "counts": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}}, open(out, "w"), indent=2)
