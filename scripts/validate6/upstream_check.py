"""Run FlashTP's unmodified upstream example/flashTP_test.py, wrapping utils.compare to also report
max abs / max rel error and the upstream verdict per comparison. Usage (same argv as upstream):
  python -I flashtp_fp64_check.py <flashTP_test.py> <ir_config> <layer_idx> <edges> fp64 False <channel_mul>"""
import sys, runpy, json, os, torch, flashTP_e3nn, flashTP_e3nn.utils as U
NAMES = ["out", "d_in1", "d_in2", "d_weight", "dd_in1", "dd_in2", "dd_weight"]
orig, rows = U.compare, []
def compare(a, b):
    name = NAMES[len(rows)] if len(rows) < len(NAMES) else f"cmp{len(rows)}"
    d = (a - b).abs(); denom = b.abs().clamp_min(1e-300)
    upstream_bad = bool(((~torch.isclose(a, b)) & (d > 1e-4)).any())
    r = dict(name=name, shape=list(a.shape), dtype=str(a.dtype), max_abs=float(d.max()), max_ref=float(b.abs().max()),
             max_rel_where_ref_gt_1em6=float((d / denom)[b.abs() > 1e-6].max()) if (b.abs() > 1e-6).any() else 0.0,
             n_not_isclose_default=int((~torch.isclose(a, b)).sum()), upstream_pass=not upstream_bad,
             strict_pass_fp64=bool(torch.allclose(a, b, rtol=1e-10, atol=1e-10)))
    rows.append(r); print(f"[{name}] {r}"); orig(a, b)
U.compare = compare
flashTP_e3nn.utils.compare = compare
script, args = sys.argv[1], sys.argv[2:]
sys.argv = [script] + args
torch.cuda.synchronize(); runpy.run_path(script, run_name="__main__"); torch.cuda.synchronize()
out = dict(args=args, flashtp_file=flashTP_e3nn.__file__, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
           upstream_tolerance="fail iff not torch.isclose(rtol=1e-5,atol=1e-8) AND |diff|>1e-4", results=rows,
           all_upstream_pass=all(r["upstream_pass"] for r in rows) and len(rows) == 7)
tag = f"{os.path.basename(args[0]).split('.')[0]}_L{args[1]}_E{args[2]}_{args[3]}_x{args[5]}"
json.dump(out, open(os.path.join(os.environ["R"], "results", "validate6", f"upstream_fp64check_graphcap_{tag}.json"), "w"), indent=2)
print("SUMMARY", tag, "all_upstream_pass=", out["all_upstream_pass"])
