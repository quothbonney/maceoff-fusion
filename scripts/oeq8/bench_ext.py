"""External conv backends (LunaTP in venv-luna, OpenEquivariance in venv-oeq) on captured MACE-OFF inputs, fp64.
Reference: MACE conv_tp (e3nn) + gather + scatter_sum, recomputed in this venv (same e3nn 0.4.4 / mace 0.3.17 / torch).
Variants per backend:
  *_full   : adapter as it would sit in MACE (edges arrive src-sorted): dst sort + graph build + gather of y,w + kernel
  *_kernel : graph built and y,w pre-permuted outside the timed region (kernel + autograd only); grads mapped back
Usage: python -I bench_ext.py luna|oeq <out_json> [systems...]"""
import os, sys, json, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import common as C

backend = sys.argv[1]; out_path = sys.argv[2]; names = sys.argv[3:] or C.SYSTEMS
calc = C.load_calc(); model = calc.models[0]
res = dict(env=C.env_info(), backend=backend, results={}, build={})


def dst_sort(ei):
    perm = torch.argsort(ei[1], stable=True)
    return perm, ei[0][perm], ei[1][perm]


if backend == "luna":
    import LunaEquivariance as LE
    res["lib"] = LE.__file__

    def build(ct):
        try:
            conv = LE.TensorProductConv(ct, dtype=torch.float64)
        except TypeError:
            conv = LE.TensorProductConv(ct)
        return conv.cuda()

    def variants(conv, N, ei):
        out = {}

        def full(x, y, w):
            perm, s, d = dst_sort(ei)
            gph = LE.make_graph(s, d, N)
            return conv(x, y[perm], w[perm], graph=gph)
        out["luna_full"] = (full, None)
        perm, s, d = dst_sort(ei)
        gph = LE.make_graph(s, d, N)
        out["luna_kernel"] = ((lambda x, y, w: conv(x, y, w, graph=gph)), perm)
        return out

elif backend == "oeq":
    import openequivariance as oeq
    res["lib"] = oeq.__file__

    def build(ct):
        insts = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in ct.instructions]
        tpp = oeq.TPProblem(ct.irreps_in1, ct.irreps_in2, ct.irreps_out, insts, shared_weights=False,
                            internal_weights=False, irrep_dtype=oeq.torch_to_oeq_dtype(torch.float64),
                            weight_dtype=oeq.torch_to_oeq_dtype(torch.float64))
        return dict(atomic=oeq.TensorProductConv(tpp, deterministic=False),
                    det=oeq.TensorProductConv(tpp, deterministic=True))

    def variants(convs, N, ei):
        out = {}
        a = convs["atomic"]
        # exactly MACE's with_oeq_conv_fusion call: rows=receiver, cols=sender, original (src-sorted) order
        out["oeq_atomic"] = ((lambda x, y, w: a(x, y, w, ei[1], ei[0])), None)
        dt = convs["det"]

        def det_full(x, y, w):
            key = ei[1] * N + ei[0]
            perm = torch.argsort(key, stable=True)
            rows, cols = ei[1][perm], ei[0][perm]
            sp = torch.argsort(cols * N + rows, stable=True)
            return dt(x, y[perm], w[perm], rows, cols, sp)
        out["oeq_det_full"] = (det_full, None)
        key = ei[1] * N + ei[0]
        perm = torch.argsort(key, stable=True)
        rows, cols = ei[1][perm].contiguous(), ei[0][perm].contiguous()
        sp = torch.argsort(cols * N + rows, stable=True)
        out["oeq_det_kernel"] = ((lambda x, y, w: dt(x, y, w, rows, cols, sp)), perm)
        return out
else:
    raise SystemExit(backend)

convs = {}
for i, blk in enumerate(model.interactions):
    try:
        convs[i] = build(blk.conv_tp)
        res["build"][f"L{i}"] = "ok"
    except Exception:
        res["build"][f"L{i}"] = traceback.format_exc()[-2500:]
        print(res["build"][f"L{i}"], flush=True)
json.dump(res, open(out_path, "w"), indent=1)

# E=0 edge case (N=3)
for i in convs:
    try:
        ct = model.interactions[i].conv_tp
        x = torch.randn(3, ct.irreps_in1.dim, dtype=torch.float64, device="cuda")
        y = torch.zeros(0, ct.irreps_in2.dim, dtype=torch.float64, device="cuda")
        w = torch.zeros(0, ct.weight_numel, dtype=torch.float64, device="cuda")
        ei = torch.zeros(2, 0, dtype=torch.int64, device="cuda")
        g = torch.randn(3, ct.irreps_out.dim, dtype=torch.float64, device="cuda")
        r0 = {}
        for vn, (fn, perm) in variants(convs[i], 3, ei).items():
            try:
                o = C.fwd_bwd(fn, x, y, w, g)
                r0[vn] = dict(out_abs_max=o[0].abs().max().item(), gx_abs_max=o[1].abs().max().item(),
                              shapes=[list(t.shape) for t in o])
            except Exception as e:
                r0[vn] = dict(error=repr(e)[:400]); torch.cuda.synchronize()
        res.setdefault("E0", {})[f"L{i}"] = r0
    except Exception:
        res.setdefault("E0", {})[f"L{i}"] = traceback.format_exc()[-800:]
print("E0", res.get("E0"), flush=True)

for name in names:
    for i in convs:
        d = C.load_cap(name, i); x, y, w, ei = d["x"], d["y"], d["w"], d["ei"]
        N = x.shape[0]
        ct = model.interactions[i].conv_tp
        g = C.gen_grad((N, ct.irreps_out.dim), seed=1000 + i)
        key = f"{name}_L{i}"; r = res["results"][key] = dict(N=N, E=ei.shape[1])
        ref = C.fwd_bwd(lambda a, b_, c: C.ref_conv(ct, a, b_, c, ei), x, y, w, g)
        try:
            vs = variants(convs[i], N, ei)
        except Exception:
            r["error"] = traceback.format_exc()[-2000:]; print(key, r["error"]); continue
        for vn, (fn, perm) in vs.items():
            rr = r[vn] = {}
            try:
                yy, ww = (y, w) if perm is None else (y[perm].contiguous(), w[perm].contiguous())
                o = C.fwd_bwd(fn, x, yy, ww, g)
                o2 = C.fwd_bwd(fn, x, yy, ww, g)
                rr["run_to_run_bitwise"] = all(torch.equal(a, b_) for a, b_ in zip(o, o2))
                if perm is not None:  # map edge grads back to original edge order
                    inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=perm.device)
                    o = (o[0], o[1], o[2][inv], o[3][inv])
                rr["err"] = {k: C.err(a, b_) for k, a, b_ in zip(("out", "gx", "gy", "gw"), o, ref)}
                rr["time"] = C.bench_conv(fn, x, yy, ww, g)
            except Exception:
                rr["error"] = traceback.format_exc()[-2000:]
                torch.cuda.synchronize()
            print(key, vn, json.dumps({k: (v if k != "time" else {t: v[t].get("median", v[t]) for t in v})
                                       for k, v in rr.items()}, default=str)[:1000], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
json.dump(res, open(out_path, "w"), indent=1)
print("done")
