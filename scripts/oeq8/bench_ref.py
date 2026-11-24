"""venv-cueq: e3nn conv (reference), FlashTP-FP64CG graphcap conv, cuEquivariance fused conv (MACE conv_fusion,
mul_ir, O3_e3nn) on captured inputs. Correctness vs e3nn fp64 reference + conv-only timing (eager, CUDA graph).
Usage: python -I bench_ref.py <out_json> [systems...]"""
import os, sys, json, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import flashtp_adapter_gc as FA  # must precede other flashTP_e3nn imports
import torch
import common as C

out_path = sys.argv[1]; args = sys.argv[2:]
ONLY = None
if args and args[0].startswith("--only="):
    ONLY = args[0][7:].split(","); args = args[1:]
names = args or C.SYSTEMS
calc = C.load_calc(); model = calc.models[0]
res = dict(env=C.env_info(), flashTP_e3nn=FA.flashTP_e3nn.__file__, results={})


def make_cueq(conv_tp):
    from mace.modules.wrapper_ops import TensorProduct, CuEquivarianceConfig
    before = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        cfg = CuEquivarianceConfig(enabled=True, layout="mul_ir", group="O3_e3nn", optimize_channelwise=True,
                                   conv_fusion=True)
        insts = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in conv_tp.instructions]
        m = TensorProduct(conv_tp.irreps_in1, conv_tp.irreps_in2, conv_tp.irreps_out, insts, shared_weights=False,
                          internal_weights=False, cueq_config=cfg).cuda()
    finally:
        torch.set_default_dtype(before)
    return m


backends = {}
for i, blk in enumerate(model.interactions):
    ct = blk.conv_tp
    b = {"e3nn": lambda x, y, w, ei, ct=ct: C.ref_conv(ct, x, y, w, ei)}
    fc = FA.FlashConv(ct, torch.float64, "cuda", check_indices=False)
    b["flashtp_gc"] = lambda x, y, w, ei, fc=fc: fc(x, y, w, ei)
    try:
        cq = make_cueq(ct)
        res.setdefault("cueq_info", {})[f"L{i}"] = dict(cls=type(cq).__name__, weight_numel=int(cq.weight_numel),
                                                        method=str(getattr(cq.conv_tp, "method", None)))
        import cuequivariance as cue, cuequivariance_torch as cuet
        from mace.tools.cg import O3_e3nn
        tin = cuet.TransposeIrrepsLayout(cue.Irreps(O3_e3nn, str(ct.irreps_in1)), source=cue.mul_ir, target=cue.ir_mul,
                                         use_fallback=True).cuda()
        tout = cuet.TransposeIrrepsLayout(cue.Irreps(O3_e3nn, str(ct.irreps_out)), source=cue.ir_mul,
                                          target=cue.mul_ir, use_fallback=True).cuda()
        # MACE uses cueq conv_fusion only with ir_mul node layout: transpose in/out (exact copies)
        b["cueq_fused_irmul_T"] = lambda x, y, w, ei, cq=cq, tin=tin, tout=tout: tout(cq(tin(x), y, w, ei))
        b["cueq_fused_raw"] = lambda x, y, w, ei, cq=cq: cq(x, y, w, ei)
    except Exception as e:
        res.setdefault("cueq_info", {})[f"L{i}"] = dict(error=traceback.format_exc()[-1500:])
    backends[i] = b

for name in names:
    for i in range(len(model.interactions)):
        d = C.load_cap(name, i); x, y, w, ei = d["x"], d["y"], d["w"], d["ei"]
        g = C.gen_grad((x.shape[0], model.interactions[i].conv_tp.irreps_out.dim), seed=1000 + i)
        key = f"{name}_L{i}"; r = res["results"][key] = dict(N=x.shape[0], E=ei.shape[1])
        ref = C.fwd_bwd(lambda a, b_, c: C.ref_conv(model.interactions[i].conv_tp, a, b_, c, ei), x, y, w, g)
        for bn, fn in backends[i].items():
            if ONLY and bn not in ONLY: continue
            rr = r[bn] = {}
            try:
                with C.quiet():
                    o = C.fwd_bwd(lambda a, b_, c: fn(a, b_, c, ei), x, y, w, g)
                    o2 = C.fwd_bwd(lambda a, b_, c: fn(a, b_, c, ei), x, y, w, g)
                rr["err"] = {k: C.err(a, b_) for k, a, b_ in zip(("out", "gx", "gy", "gw"), o, ref)}
                rr["run_to_run_bitwise"] = all(torch.equal(a, b_) for a, b_ in zip(o, o2))
                with C.quiet():
                    rr["time"] = C.bench_conv(lambda a, b_, c: fn(a, b_, c, ei), x, y, w, g)
            except Exception:
                rr["error"] = traceback.format_exc()[-2000:]
                torch.cuda.synchronize()
            print(key, bn, json.dumps({k: (v if k != "time" else {t: v[t].get("median", v[t]) for t in v})
                                       for k, v in rr.items()}, default=str)[:900], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
json.dump(res, open(out_path, "w"), indent=1)
print("done")
