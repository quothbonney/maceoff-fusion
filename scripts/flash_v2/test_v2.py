"""Tests for flashtp_adapter_v2 / hybrid_v2.  Usage: test_v2.py {core|extra} {float32|float64} out.json
core : CG precision (item 1), layer parity incl. zero-edge grads (item 2), whole-model zero-edge graphs + batches (2),
       interleaved graphs / stale-index (4), whole-model parity e3nn vs flash(sc=e3nn) vs hybrid (6).
extra: allowlist/dtype/device/int32/index checks, install/uninstall (3), state_dict / torch.save / deepcopy (5).
The global default dtype is set by THIS script (argv[2]) before anything is built; build_calc must not change it.
"""
import os, sys, json, time, copy, io, subprocess, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hybrid_v2 as HB  # noqa: E402
FA = HB.FA
import torch  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(1, f"{HB.R}/scripts/flash_proto")
from systems import get_system  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402

MODE, DEFAULT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_default_dtype({"float32": torch.float32, "float64": torch.float64}[DEFAULT])
DEF = torch.get_default_dtype()
RES = dict(mode=MODE, global_default_dtype=str(DEF), slurm_job=os.environ.get("SLURM_JOB_ID"),
           gpu=subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
           torch=torch.__version__, flashTP_e3nn=HB.flashTP_e3nn.__file__, tests={})
FAIL = []


def save():
    RES["failures"] = FAIL
    RES["pass"] = not FAIL
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1, default=str)


def record(name, fn):
    t = time.time()
    try:
        r = fn() or {}
        ok = r.pop("_ok", True)
        r["status"] = "ok" if ok else "FAIL"
    except Exception as e:  # noqa
        r = dict(status="FAIL", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        ok = False
    r["sec"] = round(time.time() - t, 2)
    if not ok:
        FAIL.append(name)
    RES["tests"][name] = r
    print(name, r["status"], {k: v for k, v in r.items() if k not in ("tb",)} if not ok else "", flush=True)
    save()
    return r


def check_default():
    assert torch.get_default_dtype() == DEF, f"global default changed to {torch.get_default_dtype()}"


# ------------------------------------------------------------------------------------------------ builders
CALCS = {}


def calc(variant):
    if variant not in CALCS:
        conv, sc = {"e3nn": ("e3nn", "e3nn"), "flash": ("flashtp", "e3nn"), "hybrid": ("flashtp", "cueq")}[variant]
        c, info = HB.build_calc(HB.MODEL, conv=conv, sc=sc, dtype=torch.float64)
        check_default()
        CALCS[variant] = (c, info)
    return CALCS[variant][0]


def ef(c, atoms):
    a = atoms.copy(); a.calc = c
    return a.get_potential_energy(), a.get_forces()


def cmp(Er, Fr, Et, Ft, n):
    d = np.abs(Ft - Fr)
    return dict(dE_per_atom=abs(Et - Er) / n, max_dF=float(d.max()) if d.size else 0.0,
                rms_dF=float(np.sqrt((d ** 2).mean())) if d.size else 0.0, E_ref=Er, E_test=Et)


# ------------------------------------------------------------------------------------------------ core tests
def t_cg_build():
    c = calc("flash")
    rep = FA.flash_cg_reports(c.models[0])
    ok = all(v["cg_tensor_dtype"] == "torch.float64" and v["cg_not_fp32_representable"] > 0
             and v["max_abs_e3nn_fp64_cg_to_table"] < 1e-14 for v in rep.values())
    return dict(cg=rep, info_cueq=calc("hybrid") and CALCS["hybrid"][1]["cueq_sc"],
                backend_map_hybrid=CALCS["hybrid"][1]["backend_map"], default_after=str(torch.get_default_dtype()),
                _ok=ok and torch.get_default_dtype() == DEF)


def e3nn_conv(block, nf, ea, w, ei):
    mji = block.conv_tp(nf[ei[0]], ea, w)
    from mace.tools.scatter import scatter_sum
    return scatter_sum(src=mji, index=ei[1], dim=0, dim_size=nf.shape[0])


def rand_inputs(fc, n, e, seed, ei=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(device="cuda", dtype=torch.float64, generator=g)
    nf = torch.randn(n, fc.in1_dim, **kw).requires_grad_()
    ea = torch.randn(e, fc.in2_dim, **kw).requires_grad_()
    w = torch.randn(e, fc.weight_numel, **kw).requires_grad_()
    if ei is None:
        ei = (torch.randint(0, n, (2, e), device="cuda", generator=g) if e > 0
              else torch.empty(2, 0, dtype=torch.int64, device="cuda"))
    return nf, ea, w, ei


def layer_parity(fc, orig, nf, ea, w, ei):
    blk = type("B", (), {"conv_tp": orig})
    outs = {}
    for tag, f in (("flash", lambda: fc(nf, ea, w, ei)), ("e3nn", lambda: e3nn_conv(blk, nf, ea, w, ei))):
        o = f()
        go = torch.ones_like(o) if o.numel() == 0 else torch.cos(torch.arange(o.numel(), device="cuda", dtype=o.dtype)).view_as(o)
        gs = torch.autograd.grad(o, (nf, ea, w), go, allow_unused=True)
        outs[tag] = [o] + [torch.zeros_like(x) if gg is None else gg for gg, x in zip(gs, (nf, ea, w))]
    r = {}
    for k, a, b in zip(["out", "d_node_feats", "d_edge_attrs", "d_tp_weights"], outs["flash"], outs["e3nn"]):
        assert a.shape == b.shape, (k, a.shape, b.shape)
        den = b.abs().max().item() if b.numel() else 0.0
        mx = (a - b).abs().max().item() if a.numel() else 0.0
        r[k] = dict(shape=list(a.shape), max_abs=mx, rel=mx / den if den else None, ref_absmax=den)
    return r


def t_layer():
    m = calc("flash").models[0]
    res, ok = {}, True
    for li, b in enumerate(m.interactions):
        fc = b.conv_tp
        for tag, (n, e) in {"N64_E2048": (64, 2048), "N10_E0": (10, 0), "N1_E0": (1, 0), "N0_E0": (0, 0)}.items():
            r = layer_parity(fc, fc.orig, *rand_inputs(fc, n, e, 1000 + li))
            res[f"L{li}_{tag}"] = r
            for k, v in r.items():
                if e == 0:
                    ok &= v["max_abs"] == 0.0
                else:
                    ok &= v["rel"] is not None and v["rel"] < 1e-13
    return dict(layers=res, _ok=ok)


def t_layer_interleaved():
    """Same FlashConv called with changing graphs; same E different pairs (incl. freed-and-reallocated index tensors)."""
    m = calc("flash").models[0]
    fc = m.interactions[1].conv_tp
    worst, rows = 0.0, []
    n, e = 200, 3000
    base = rand_inputs(fc, n, e, 7)
    for step in range(8):
        g = torch.Generator(device="cuda").manual_seed(step)
        if step % 2 == 0:  # same E, different pairs, index tensor freed and reallocated (allocator address reuse)
            ei = torch.randint(0, n, (2, e), device="cuda", generator=g)
            nf, ea, w = base[0], base[1], base[2]
        else:
            nf, ea, w, ei = rand_inputs(fc, n, 500 + 300 * step, 100 + step)
        r = layer_parity(fc, fc.orig, nf, ea, w, ei)
        worst = max(worst, max(v["rel"] or 0.0 for v in r.values()))
        rows.append(dict(step=step, E=int(ei.shape[1]), ptr=ei.data_ptr(), out_rel=r["out"]["rel"]))
        del ei
    # in-place modification of the same index tensor (same address, same E, same _version semantics irrelevant)
    nf, ea, w, ei = rand_inputs(fc, n, e, 9)
    r1 = layer_parity(fc, fc.orig, nf, ea, w, ei)
    ei[1] = ei[1].flip(0)
    r2 = layer_parity(fc, fc.orig, nf, ea, w, ei)
    worst = max(worst, r1["out"]["rel"], r2["out"]["rel"])
    return dict(rows=rows, inplace=[r1["out"]["rel"], r2["out"]["rel"]], worst_rel=worst, _ok=worst < 1e-13)


def two_far(sym="OO", d=20.0):
    return Atoms(sym, positions=[[0, 0, 0], [d, 0, 0]])


ZERO_SYSTEMS = {
    "single_O": lambda: Atoms("O", positions=[[0, 0, 0]]),
    "single_H": lambda: Atoms("H", positions=[[0, 0, 0]]),
    "two_O_20A": lambda: two_far("OO"),
    "four_atoms_all_gt_rmax": lambda: Atoms("OHCN", positions=[[0, 0, 0], [6, 0, 0], [0, 7, 0], [0, 0, 8]]),
    "mixed_isolatedO_plus_H2O": lambda: molecule("H2O") + Atoms("O", positions=[[15, 0, 0]]),
}


def t_zero_edge_model():
    res, ok = {}, True
    for name, f in ZERO_SYSTEMS.items():
        at = f()
        Er, Fr = ef(calc("e3nn"), at)
        row = {}
        for v in ("flash", "hybrid"):
            Et, Ft = ef(calc(v), at)
            r = cmp(Er, Fr, Et, Ft, len(at))
            r["finite"] = bool(np.isfinite(Et) and np.isfinite(Ft).all())
            r["max_abs_F"] = float(np.abs(Ft).max())
            row[v] = r
            ok &= r["finite"] and r["dE_per_atom"] < 1e-12 and r["max_dF"] < 1e-12
            if name != "mixed_isolatedO_plus_H2O":
                ok &= r["max_abs_F"] == 0.0
        res[name] = row
    return dict(systems=res, _ok=ok)


def model_batch(c, atoms_list):
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools
    with torch_tools.default_dtype(c.default_dtype):
        ds = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a), z_table=c.z_table, cutoff=c.r_max,
                                               heads=c.available_heads) for a in atoms_list]
        b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds), shuffle=False, drop_last=False)))
        b = b.to("cuda")
        out = c.models[0](b.to_dict(), training=False, compute_force=True)
    return out["energy"].detach().cpu().numpy(), out["forces"].detach().cpu().numpy(), int(b["edge_index"].shape[1])


def t_zero_edge_batch():
    batches = {"single_O+single_H (all E=0)": [ZERO_SYSTEMS["single_O"](), ZERO_SYSTEMS["single_H"]()],
               "single_O+H2O+two_O_20A": [ZERO_SYSTEMS["single_O"](), molecule("H2O"), two_far()],
               "H2O+single_H": [molecule("H2O"), ZERO_SYSTEMS["single_H"]()]}
    res, ok = {}, True
    for name, al in batches.items():
        Er, Fr, ne = model_batch(calc("e3nn"), al)
        row = dict(n_edges=ne)
        for v in ("flash", "hybrid"):
            Et, Ft, _ = model_batch(calc(v), al)
            row[v] = dict(max_dE=float(np.abs(Et - Er).max()), max_dF=float(np.abs(Ft - Fr).max()),
                          E_ref=Er.tolist(), E_test=Et.tolist())
            ok &= row[v]["max_dE"] < 1e-10 and row[v]["max_dF"] < 1e-12
        res[name] = row
    return dict(batches=res, _ok=ok)


def perturbed(at, seed, scale=0.03):
    a = at.copy()
    a.positions = a.positions + np.random.default_rng(seed).normal(scale=scale, size=a.positions.shape)
    return a


def t_parity():
    res, ok = {}, True
    for sysname in ("H2O", "water192", "water648"):
        base = get_system(sysname)
        n = len(base)
        rows = []
        for k in range(3):  # changing coordinates
            at = base if k == 0 else perturbed(base, 100 + k)
            Er, Fr = ef(calc("e3nn"), at)
            Er2, Fr2 = ef(calc("e3nn"), at)
            row = dict(geom=k, e3nn_repeat=cmp(Er, Fr, Er2, Fr2, n))
            for v in ("flash", "hybrid"):
                Et, Ft = ef(calc(v), at)
                Et2, Ft2 = ef(calc(v), at)
                row[v] = cmp(Er, Fr, Et, Ft, n)
                row[v + "_repeat"] = cmp(Et, Ft, Et2, Ft2, n)
            rows.append(row)
        agg = {}
        for v in ("flash", "hybrid", "flash_repeat", "hybrid_repeat", "e3nn_repeat"):
            agg[v] = {m: max(r[v][m] for r in rows) for m in ("dE_per_atom", "max_dF", "rms_dF")}
        res[sysname] = dict(n_atoms=n, worst=agg, rows=rows)
        for v in ("flash", "hybrid"):
            ok &= agg[v]["dE_per_atom"] < 1e-11 and agg[v]["max_dF"] < 1e-11
    return dict(systems=res, _ok=ok)


def t_interleaved_model():
    """Interleave graphs of different and of identical edge counts through one calculator (no stale indices)."""
    w = get_system("water192")
    perm = np.random.default_rng(5).permutation(len(w))
    wp = w[perm]  # same edge count, different (src,dst) pairs
    seq = [("H2O", molecule("H2O")), ("water192", w), ("single_O", ZERO_SYSTEMS["single_O"]()), ("water192_perm", wp),
           ("water192", w), ("H2O_pert", perturbed(molecule("H2O"), 3)), ("water192_perm", wp), ("water192_pert", perturbed(w, 4))]
    rows, ok = [], True
    for v in ("flash", "hybrid"):
        for name, at in seq:
            Er, Fr = ef(calc("e3nn"), at)
            Et, Ft = ef(calc(v), at)
            r = cmp(Er, Fr, Et, Ft, len(at)); r.update(variant=v, system=name)
            rows.append(r)
            ok &= r["dE_per_atom"] < 1e-11 and r["max_dF"] < 1e-11
    return dict(rows=rows, _ok=ok)


# ------------------------------------------------------------------------------------------------ extra tests
def expect_error(fn, contains=None):
    try:
        fn()
    except FA.AdapterError as e:
        return dict(raised="AdapterError", msg=str(e)[:300], _ok=(contains is None or contains in str(e)))
    except Exception as e:  # noqa
        return dict(raised=type(e).__name__, msg=str(e)[:300], _ok=False)
    return dict(raised=None, _ok=False)


def fresh_model():
    from mace.calculators import MACECalculator
    c = MACECalculator(model_paths=HB.MODEL, device="cuda", default_dtype="float64", compute_stress=False)
    torch.set_default_dtype(DEF)
    return c.models[0]


def t_checks():
    out = {}
    m = fresh_model()
    # allowlist: subclass, Magnetic-named class, cueq model, already fused, double install
    b0 = m.interactions[0]
    orig_cls = b0.__class__
    b0.__class__ = type("MyInteraction", (orig_cls,), {})
    out["reject_subclass"] = expect_error(lambda: FA.install(m), "unsupported interaction class")
    b0.__class__ = type("MagneticInteractionBlock", (orig_cls,), {})
    out["reject_magnetic"] = expect_error(lambda: FA.install(m), "magnetic")
    b0.__class__ = orig_cls
    out["no_partial_install_after_reject"] = dict(installed=FA.is_installed(m), _ok=not FA.is_installed(m)
                                                  and not any(hasattr(b, "conv_fusion") for b in m.interactions))
    b0.conv_fusion = True
    out["reject_already_fused_flag"] = expect_error(lambda: FA.install(m), "already fused")
    del b0.conv_fusion
    from mace.calculators import MACECalculator
    cq = MACECalculator(model_paths=HB.MODEL, device="cuda", default_dtype="float64", compute_stress=False, enable_cueq=True)
    torch.set_default_dtype(DEF)
    out["reject_cueq_model"] = expect_error(lambda: FA.install(cq.models[0]))
    del cq
    FA.install(m)
    out["double_install"] = expect_error(lambda: FA.install(m), "already installed")
    fc = m.interactions[1].conv_tp
    nf, ea, w, ei = rand_inputs(fc, 16, 40, 3)
    out["dtype_mismatch_fp32"] = expect_error(lambda: fc(nf.float(), ea.float(), w.float(), ei), "dtype")
    out["dtype_mismatch_one_input"] = expect_error(lambda: fc(nf, ea.float(), w, ei), "dtype")
    out["device_mismatch_cpu"] = expect_error(lambda: fc(nf.cpu(), ea.cpu(), w.cpu(), ei.cpu()), "device")
    out["edge_index_cpu"] = expect_error(lambda: fc(nf, ea, w, ei.cpu()), "device")
    out["edge_index_float"] = expect_error(lambda: fc(nf, ea, w, ei.double()), "dtype")
    out["shape_mismatch"] = expect_error(lambda: fc(nf, ea[:-1], w, ei), "shape")
    bad = ei.clone(); bad[0, 3] = 16
    out["index_out_of_range"] = expect_error(lambda: fc(nf, ea, w, bad), "out of range")
    bad2 = ei.clone(); bad2[1, 0] = -1
    out["index_negative"] = expect_error(lambda: fc(nf, ea, w, bad2), "out of range")
    big = 2**31
    nf_big = torch.zeros(1, fc.in1_dim, device="cuda", dtype=torch.float64).expand(big, fc.in1_dim)
    out["n_nodes_ge_2^31"] = expect_error(lambda: fc(nf_big, ea, w, ei), "int32")
    ei_big = torch.zeros(2, 1, device="cuda", dtype=torch.int64).expand(2, big)
    ea_big = torch.zeros(1, fc.in2_dim, device="cuda", dtype=torch.float64).expand(big, fc.in2_dim)
    w_big = torch.zeros(1, fc.weight_numel, device="cuda", dtype=torch.float64).expand(big, fc.weight_numel)
    out["n_edges_ge_2^31"] = expect_error(lambda: fc(nf, ea_big, w_big, ei_big), "int32")
    ok_call = fc(nf, ea, w, ei.to(torch.int32))
    out["int32_edge_index_accepted"] = dict(shape=list(ok_call.shape))
    FA.uninstall(m)
    # dtype mismatch at install: fp32 model
    m32 = fresh_model().float()
    out["reject_model_dtype_mismatch"] = expect_error(lambda: FA.install(m32, dtype=torch.float64), "dtype")
    out["fp32_model_install_ok"] = dict(n=len(FA.install(m32)), cg=FA.flash_cg_reports(m32)["interactions.0"]["cg_tensor_dtype"])
    FA.uninstall(m32)
    out["default_unchanged"] = dict(_ok=torch.get_default_dtype() == DEF)
    return dict(checks=out, _ok=all(v.get("_ok", True) for v in out.values()))


def sd_cpu(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def t_install_state():
    out = {}
    m = fresh_model()
    mods0 = {n: id(x) for n, x in m.named_modules()}
    sd0 = sd_cpu(m)
    keys0 = list(m.state_dict().keys())
    md0 = sorted(m.state_dict()._metadata.keys())
    FA.install(m)
    sd1 = m.state_dict()
    out["state_dict_installed_keys"] = dict(same_keys_and_order=list(sd1.keys()) == keys0,
                                            values_bitwise=all(torch.equal(sd1[k].cpu(), sd0[k]) for k in keys0),
                                            same_metadata_keys=sorted(sd1._metadata.keys()) == md0,
                                            _ok=list(sd1.keys()) == keys0 and sorted(sd1._metadata.keys()) == md0
                                            and all(torch.equal(sd1[k].cpu(), sd0[k]) for k in keys0))
    # load_state_dict while installed (strict) round-trips through the hooks
    r = m.load_state_dict(sd0, strict=True)
    out["load_state_dict_installed_strict"] = dict(missing=r.missing_keys, unexpected=r.unexpected_keys,
                                                   _ok=not r.missing_keys and not r.unexpected_keys)
    # save state_dict while installed -> load into a fresh stock model strictly
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
    m2 = fresh_model()
    r2 = m2.load_state_dict(torch.load(buf, weights_only=False), strict=True)
    out["state_dict_saved_installed_loads_into_stock"] = dict(_ok=not r2.missing_keys and not r2.unexpected_keys)
    del m2
    out["torch_save_model_installed_raises"] = expect_error(lambda: torch.save(m, io.BytesIO()), "uninstall")
    out["deepcopy_installed_raises"] = expect_error(lambda: copy.deepcopy(m), "uninstall")
    out["install_attr_not_in_state_dict"] = dict(_ok=not any("flashtp" in k.lower() or "ftp" in k for k in m.state_dict()))
    u1 = FA.uninstall(m); u2 = FA.uninstall(m); u3 = FA.uninstall(m)
    mods1 = {n: id(x) for n, x in m.named_modules()}
    sd2 = m.state_dict()
    out["uninstall_restore"] = dict(first=u1, second=u2, third=u3, same_module_objects=mods1 == mods0,
                                    same_keys=list(sd2.keys()) == keys0,
                                    values_bitwise=all(torch.equal(sd2[k].cpu(), sd0[k]) for k in keys0),
                                    no_conv_fusion=not any(hasattr(b, "conv_fusion") for b in m.interactions),
                                    _ok=u1 and not u2 and not u3 and mods1 == mods0 and list(sd2.keys()) == keys0
                                    and all(torch.equal(sd2[k].cpu(), sd0[k]) for k in keys0)
                                    and not any(hasattr(b, "conv_fusion") for b in m.interactions))
    buf = io.BytesIO(); torch.save(m, buf)
    out["torch_save_after_uninstall_ok"] = dict(bytes=buf.tell())
    # reinstall after uninstall works and gives same E/F as e3nn
    FA.install(m)
    out["reinstall_ok"] = dict(installed=FA.is_installed(m))
    FA.uninstall(m)
    # hybrid: cueq double install / idempotent uninstall
    c, _ = HB.build_calc(HB.MODEL, conv="flashtp", sc="cueq")
    hm = c.models[0]
    out["hybrid_double_install_sc"] = expect_error(lambda: HB.install_cueq_sc(hm), "already")
    out["hybrid_double_install_flash"] = expect_error(lambda: FA.install(hm), "already")
    out["hybrid_uninstall_idempotent"] = dict(flash=[FA.uninstall(hm), FA.uninstall(hm)],
                                              sc=[HB.uninstall_cueq_sc(hm), HB.uninstall_cueq_sc(hm)])
    out["hybrid_uninstall_idempotent"]["_ok"] = out["hybrid_uninstall_idempotent"]["flash"] == [True, False] and \
        out["hybrid_uninstall_idempotent"]["sc"] == [True, False]
    out["hybrid_uninstalled_keys_stock"] = dict(_ok=list(hm.state_dict().keys()) == keys0)
    # E/F after uninstall vs e3nn reference calc
    at = get_system("water192")
    a = at.copy(); a.calc = c
    Eu, Fu = a.get_potential_energy(), a.get_forces()
    Er, Fr = ef(calc("e3nn"), at)
    out["hybrid_after_uninstall_vs_e3nn"] = cmp(Er, Fr, Eu, Fu, len(at))
    return dict(checks=out, _ok=all(v.get("_ok", True) for v in out.values()))


if MODE == "core":
    record("cg_build_fp64", t_cg_build)
    record("layer_parity_and_zero_edge_grads", t_layer)
    record("layer_interleaved_graphs", t_layer_interleaved)
    record("zero_edge_whole_model", t_zero_edge_model)
    record("zero_edge_batches", t_zero_edge_batch)
    record("interleaved_whole_model", t_interleaved_model)
    record("whole_model_parity", t_parity)
else:
    record("checks", t_checks)
    record("install_state", t_install_state)
check_default()
save()
print("PASS" if not FAIL else f"FAILURES: {FAIL}")
