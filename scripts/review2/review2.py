"""Independent review (review2) of scripts/flash_v2 adapter + hybrid.
Usage: review2.py <float32|float64 global default> <out_prefix>
Writes <out_prefix>.json (findings evidence) and <out_prefix>.npz (raw outputs for cross-default comparison).
"""
import os, sys, json, time, gc, copy, io, subprocess, traceback, itertools, weakref
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_v2")
import hybrid_v2 as HB  # noqa: E402  (must come before anything imports flashTP_e3nn)
FA = HB.FA
import torch  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402
from mace import data as mace_data  # noqa: E402
from mace.tools import torch_tools, torch_geometric  # noqa: E402

DEFAULT, PREFIX = sys.argv[1], sys.argv[2]
torch.set_default_dtype({"float32": torch.float32, "float64": torch.float64}[DEFAULT])
DEF = torch.get_default_dtype()
RES = dict(global_default=str(DEF), slurm_job=os.environ.get("SLURM_JOB_ID"),
           gpu=subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(), tests={})
NPZ = {}
FAIL = []
M = HB.MODEL


def save():
    RES["failures"] = FAIL
    with open(PREFIX + ".json", "w") as f:
        json.dump(RES, f, indent=1, default=str)
    np.savez(PREFIX + ".npz", **NPZ)


def record(name, fn):
    t = time.time()
    try:
        r = fn() or {}
        ok = r.pop("_ok", True)
        r["status"] = "ok" if ok else "FAIL"
    except Exception as e:  # noqa
        r = dict(status="ERROR", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
        ok = False
    r["sec"] = round(time.time() - t, 2)
    if not ok:
        FAIL.append(name)
    RES["tests"][name] = r
    print("==", name, r["status"], json.dumps(r, default=str)[:3000], flush=True)
    save()
    return r


# ----------------------------------------------------------------------------------------------- systems
def rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def water_box(nside=6, spacing=3.104, seed=0):
    rng = np.random.default_rng(seed)
    w = molecule("H2O")
    pos = w.positions - w.positions[0]
    P, S = [], []
    for i, j, k in itertools.product(range(nside), repeat=3):
        P.append(pos @ rot(rng).T + (np.array([i, j, k]) + 0.5) * spacing + rng.normal(0, 0.1, 3))
        S += ["O", "H", "H"]
    return Atoms(S, positions=np.concatenate(P), cell=[nside * spacing] * 3, pbc=True)


def pert(a, seed, s=0.03):
    a = a.copy()
    a.positions += np.random.default_rng(seed).normal(0, s, a.positions.shape)
    return a


MOLS = {n: pert(molecule(n), 1) for n in ("CH3CH2OH", "CH3COOH", "C6H6", "CH3SH", "CH3Cl")}
WBOX = water_box(6)             # 648 atoms, periodic
WBOX_S = water_box(4, seed=2)   # 192 atoms, periodic
STRAINED = WBOX_S.copy()
STRAINED.set_cell(WBOX_S.cell[:] @ np.array([[1.02, 0.01, 0], [0.01, 0.99, 0.0], [0, 0.0, 1.01]]), scale_atoms=True)


# ----------------------------------------------------------------------------------------------- evaluation
CALCS = {}
VAR = {"e3nn": ("e3nn", "e3nn"), "flash": ("flashtp", "e3nn"), "hybrid": ("flashtp", "cueq")}


def get(variant, dtype=torch.float64):
    k = (variant, dtype)
    if k not in CALCS:
        conv, sc = VAR[variant]
        c, info = HB.build_calc(M, conv=conv, sc=sc, dtype=dtype)
        assert torch.get_default_dtype() == DEF, "build_calc changed default dtype"
        CALCS[k] = (c, info)
    return CALCS[k][0]


def make_batch(calc, atoms_list):
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    gs = []
    with torch_tools.default_dtype(calc.default_dtype):
        for a in atoms_list:
            cfg = mace_data.config_from_atoms(a, key_specification=keyspec, head_name=calc.head)
            gs.append(mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max,
                                                       heads=calc.available_heads))
    return torch_geometric.Batch.from_data_list(gs).to(calc.device)


def run(calc, atoms_list, stress=False, training=False):
    model = calc.models[0]
    b = make_batch(calc, atoms_list)
    bd = b.to_dict()
    with torch_tools.default_dtype(calc.default_dtype):
        out = model(bd, compute_stress=stress, training=training)
    E = out["energy"].detach().double().cpu().numpy()
    F = out["forces"].detach().double().cpu().numpy()
    S = out["stress"].detach().double().cpu().numpy() if (stress and out.get("stress") is not None) else None
    return E, F, S, int(bd["edge_index"].shape[1])


def cmp(ref, tst, n):
    (Er, Fr, Sr, _), (Et, Ft, St, _) = ref, tst
    d = np.abs(Ft - Fr)
    r = dict(dE_per_atom=float(np.abs(Et - Er).max() / n), max_dF=float(d.max()) if d.size else 0.0,
             rms_dF=float(np.sqrt((d ** 2).mean())) if d.size else 0.0,
             max_absF_ref=float(np.abs(Fr).max()) if Fr.size else 0.0)
    if Sr is not None and St is not None:
        r["max_dS_eV_A3"] = float(np.abs(St - Sr).max())
        r["max_absS_ref"] = float(np.abs(Sr).max())
    return r


# ----------------------------------------------------------------------------------------------- tests
def t_parity_fp64():
    out, ok = {}, True
    cases = [(n, [a], False) for n, a in MOLS.items()] + [("waterbox648_pbc", [WBOX], True),
                                                         ("waterbox192_strained_pbc", [STRAINED], True)]
    for name, al, st in cases:
        n = sum(len(a) for a in al)
        ref = run(get("e3nn"), al, stress=st)
        NPZ[f"{name}_e3nn_E"], NPZ[f"{name}_e3nn_F"] = ref[0], ref[1]
        out[name] = dict(n_atoms=n, n_edges=ref[3])
        for v in ("flash", "hybrid"):
            t = run(get(v), al, stress=st)
            NPZ[f"{name}_{v}_E"], NPZ[f"{name}_{v}_F"] = t[0], t[1]
            if t[2] is not None:
                NPZ[f"{name}_{v}_S"] = t[2]
            c = cmp(ref, t, n)
            out[name][v] = c
            if c["dE_per_atom"] > 1e-11 or c["max_dF"] > 1e-10 or c.get("max_dS_eV_A3", 0) > 1e-11:
                ok = False
        if ref[2] is not None:
            NPZ[f"{name}_e3nn_S"] = ref[2]
    # stress through the ASE calculator path (compute_stress is forced False by build_calc)
    c = get("flash")
    a = WBOX_S.copy(); a.calc = c
    try:
        a.get_stress()
        out["ase_get_stress_flash"] = "works"
    except Exception as e:  # noqa
        out["ase_get_stress_flash"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["_ok"] = ok
    return out


def t_fp32():
    out = {}
    cases = [("CH3CH2OH", [MOLS["CH3CH2OH"]], False), ("CH3SH", [MOLS["CH3SH"]], False),
             ("waterbox648_pbc", [WBOX], True)]
    for name, al, st in cases:
        n = sum(len(a) for a in al)
        r64 = run(get("e3nn"), al, stress=st)
        r32 = run(get("e3nn", torch.float32), al, stress=st)
        out[name] = {"e3nn32_vs_e3nn64": cmp(r64, r32, n)}
        for v in ("flash", "hybrid"):
            t = run(get(v, torch.float32), al, stress=st)
            out[name][f"{v}32_vs_e3nn32"] = cmp(r32, t, n)
            out[name][f"{v}32_vs_e3nn64"] = cmp(r64, t, n)
        # e3nn fp32 run-to-run noise
        r32b = run(get("e3nn", torch.float32), al, stress=st)
        out[name]["e3nn32_repeat"] = cmp(r32, r32b, n)
    out["flash32_cg"] = CALCS[("flash", torch.float32)][1]["flash_cg"]
    return out


def t_cg_tables():
    out = {}
    for v in ("flash", "hybrid"):
        m = get(v).models[0]
        for i, b in enumerate(m.interactions):
            vals = np.array([float(x) for x in b.conv_tp.ftp.unique_cg_val])
            NPZ[f"cg_{v}_L{i}"] = vals
            md = [t for t in b.conv_tp.ftp.metadata_list if torch.is_tensor(t)]
            for j, t in enumerate(md):
                NPZ[f"meta_{v}_L{i}_{j}"] = t.detach().cpu().double().numpy() if t.is_floating_point() else t.detach().cpu().numpy()
            out[f"{v}_L{i}"] = b.conv_tp.cg
    return out


def t_graph_changes():
    """One flash + one hybrid calc; alternate systems via ASE (incl. in-place edits on the same Atoms object)."""
    out, ok = {}, True
    e3 = get("e3nn")
    seqs = []
    a = WBOX_S.copy()  # the same Atoms object edited in place
    perm = np.random.default_rng(5).permutation(len(WBOX_S))
    wperm = WBOX_S[perm]  # same #edges, different edge_index
    seq = [("wbox192", lambda: WBOX_S), ("ethanol", lambda: MOLS["CH3CH2OH"]), ("wbox192_perm", lambda: wperm),
           ("wbox648", lambda: WBOX), ("ch3cl", lambda: MOLS["CH3Cl"]), ("wbox192", lambda: WBOX_S),
           ("wbox192_perm", lambda: wperm), ("strained", lambda: STRAINED)]
    for v in ("flash", "hybrid"):
        c = get(v)
        res = []
        for name, f in seq:
            at = f().copy(); at.calc = c
            E, F = at.get_potential_energy(), at.get_forces()
            r = run(e3, [f()])
            d = dict(sys=name, dE=abs(E - r[0][0]), max_dF=float(np.abs(F - r[1]).max()))
            res.append(d)
            ok &= d["dE"] < 1e-9 and d["max_dF"] < 1e-10
        # in-place edits on one Atoms object with a persistent calc
        a = WBOX_S.copy(); a.calc = c
        rng = np.random.default_rng(11)
        for k in range(6):
            if k % 2:
                a.positions[rng.integers(len(a))] += rng.normal(0, 0.4, 3)  # in-place single-atom move (edge count changes)
            else:
                a.positions += rng.normal(0, 0.02, a.positions.shape)  # in-place global edit
            E, F = a.get_potential_energy(), a.get_forces()
            r = run(e3, [a.copy()]); r2 = run(e3, [a.copy()])
            d = dict(sys=f"inplace{k}", n_edges=r[3], dE=abs(E - r[0][0]), max_dF=float(np.abs(F - r[1]).max()),
                     E_ref=float(r[0][0]), max_absF_ref=float(np.abs(r[1]).max()),
                     e3nn_self_dE=abs(r2[0][0] - r[0][0]), e3nn_self_dF=float(np.abs(r2[1] - r[1]).max()))
            res.append(d)
            ok &= d["dE"] / abs(d["E_ref"]) < 1e-12 and d["max_dF"] / d["max_absF_ref"] < 1e-11
        # in-place species change (numbers) on same object
        a.numbers[0] = 6
        E, F = a.get_potential_energy(), a.get_forces()
        r = run(e3, [a.copy()])
        res.append(dict(sys="inplace_numbers", dE=abs(E - r[0][0]), max_dF=float(np.abs(F - r[1]).max()), E_ref=float(r[0][0]),
                        max_absF_ref=float(np.abs(r[1]).max())))
        ok &= res[-1]["dE"] / abs(res[-1]["E_ref"]) < 1e-12 and res[-1]["max_dF"] / res[-1]["max_absF_ref"] < 1e-11
        # cell change in place
        a.set_cell(a.cell[:] * 1.03, scale_atoms=False)
        E, F = a.get_potential_energy(), a.get_forces()
        r = run(e3, [a.copy()])
        res.append(dict(sys="inplace_cell", dE=abs(E - r[0][0]), max_dF=float(np.abs(F - r[1]).max()), E_ref=float(r[0][0]),
                        max_absF_ref=float(np.abs(r[1]).max())))
        ok &= res[-1]["dE"] / abs(res[-1]["E_ref"]) < 1e-12 and res[-1]["max_dF"] / res[-1]["max_absF_ref"] < 1e-11
        out[v] = res
    # multi-graph batch (incl. isolated atom and far pair) vs per-graph e3nn
    far = Atoms("OC", positions=[[0, 0, 0], [0, 0, 30]])
    single = Atoms("Cl", positions=[[0, 0, 0]])
    al = [MOLS["CH3CH2OH"], single, WBOX_S, far, MOLS["CH3SH"]]
    for v in ("flash", "hybrid"):
        Eb, Fb, _, ne = run(get(v), al)
        Ebe, Fbe, _, _ = run(e3, al)
        per = [run(e3, [x])[0][0] for x in al]
        out[f"batch_{v}"] = dict(n_edges=ne, max_dE_vs_e3nn_batch=float(np.abs(Eb - Ebe).max()),
                                 max_dE_vs_e3nn_single=float(np.abs(Eb - np.array(per)).max()),
                                 max_dF=float(np.abs(Fb - Fbe).max()))
        ok &= out[f"batch_{v}"]["max_dE_vs_e3nn_single"] < 1e-9 and out[f"batch_{v}"]["max_dF"] < 1e-10
    out["_ok"] = bool(ok)
    return out


def t_zero_edge():
    out, ok = {}, True
    single = Atoms("Cl", positions=[[0, 0, 0]])
    far = Atoms("OCS", positions=[[0, 0, 0], [0, 0, 30], [30, 0, 0]])
    box1 = Atoms("O", positions=[[1, 1, 1]], cell=[20, 20, 20], pbc=True)  # periodic, zero edges -> stress path
    for name, al, st in [("single_Cl", [single], False), ("far3", [far], False), ("pbc_single_O", [box1], True),
                         ("batch_all_isolated", [single, far], False)]:
        r = {}
        ref = run(get("e3nn"), al, stress=st)
        for v in ("flash", "hybrid"):
            t = run(get(v), al, stress=st)
            r[v] = cmp(ref, t, sum(len(a) for a in al))
            r[v]["S"] = None if t[2] is None else float(np.abs(t[2]).max())
            ok &= r[v]["dE_per_atom"] < 1e-12 and r[v]["max_dF"] == 0.0
        r["n_edges"] = ref[3]
        out[name] = r
    # training=True (create_graph) + backward of a force loss to parameters: double backward through _ZeroConv
    for name, al in [("single_Cl", [single]), ("mixed_ethanol+Cl", [MOLS["CH3CH2OH"], single])]:
        r = {}
        grads = {}
        for v in ("e3nn", "flash"):
            c = get(v)
            model = c.models[0]
            b = make_batch(c, al).to_dict()
            model.zero_grad(set_to_none=True)
            rg = {k: p.requires_grad for k, p in model.named_parameters()}
            for p_ in model.parameters():
                p_.requires_grad_(True)
            try:
                with torch_tools.default_dtype(c.default_dtype):
                    o = model(b, compute_stress=False, training=True)
                loss = o["energy"].sum() + (o["forces"] ** 2).sum()
                loss.backward()
                g = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
                grads[v] = {k.replace(".orig.", "."): x for k, x in g.items()}
                r[v] = dict(ok=True, n_grads=len(g), forces_requires_grad=bool(o["forces"].requires_grad))
            except Exception as e:  # noqa
                r[v] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            model.zero_grad(set_to_none=True)
            for k, p_ in model.named_parameters():
                p_.requires_grad_(rg[k])
        if "e3nn" in grads and "flash" in grads:
            common = set(grads["e3nn"]) & set(grads["flash"])
            r["n_common"] = len(common)
            r["max_param_grad_diff"] = max([float((grads["e3nn"][k] - grads["flash"][k]).abs().max()) for k in common] or [None])
            r["keys_only_e3nn"] = sorted(set(grads["e3nn"]) - set(grads["flash"]))[:10]
            r["keys_only_flash"] = sorted(set(grads["flash"]) - set(grads["e3nn"]))[:10]
        out[f"train_{name}"] = r
    out["_ok"] = bool(ok)
    return out


def sd_snapshot(m):
    sd = m.state_dict()
    return {k: v.detach().cpu().clone() for k, v in sd.items()}, dict(getattr(sd, "_metadata", {}))


def hooks_count(m):
    n = 0
    for mod in m.modules():
        n += len(mod._state_dict_hooks) + len(mod._load_state_dict_pre_hooks) + len(getattr(mod, "_state_dict_pre_hooks", {}))
        n += len(getattr(mod, "_load_state_dict_post_hooks", {}))
    return n


def t_cleanup():
    out, ok = {}, True
    c = get("e3nn")
    m = c.models[0]
    sd0, md0 = sd_snapshot(m)
    h0 = hooks_count(m)
    convs0 = [id(b.conv_tp) for b in m.interactions]
    at = [MOLS["CH3COOH"]]
    ref = run(c, at)
    gc.collect(); torch.cuda.synchronize()
    mem = [torch.cuda.memory_allocated()]
    refs = []
    for i in range(20):
        FA.install(m, check_indices=bool(i % 2))
        fc = m.interactions[0].conv_tp
        refs.append(weakref.ref(fc))
        sdi, mdi = sd_snapshot(m)
        if i == 0:
            out["installed_sd_keys_equal"] = list(sdi) == list(sd0)
            out["installed_sd_values_bitwise"] = all(torch.equal(sdi[k], sd0[k]) for k in sd0) if list(sdi) == list(sd0) else False
            out["installed_md_keys_equal"] = sorted(mdi) == sorted(md0)
            out["installed_md_values_equal"] = all(mdi.get(k) == md0[k] for k in md0)
            t = run(c, at)
            out["installed_parity"] = cmp(ref, t, len(at[0]))
            # torch.save / deepcopy guards
            for nm, f in (("torch_save", lambda: torch.save(m, io.BytesIO())), ("deepcopy", lambda: copy.deepcopy(m))):
                try:
                    f(); out[nm] = "NO ERROR"
                except FA.AdapterError:
                    out[nm] = "AdapterError"
                except Exception as e:  # noqa
                    out[nm] = f"{type(e).__name__}"
            # load_state_dict strict while installed
            m.load_state_dict({k: v.to(c.device) for k, v in sd0.items()}, strict=True)
            out["load_sd_strict_installed"] = "ok"
            del fc
            fc = None
        else:
            run(c, at)
        del fc
        u1, u2 = FA.uninstall(m), FA.uninstall(m)
        if not (u1 is True and u2 is False):
            ok = False
            out.setdefault("uninstall_idempotency_fail", []).append((i, u1, u2))
        gc.collect(); torch.cuda.synchronize()
        mem.append(torch.cuda.memory_allocated())
    out["mem_allocated_bytes_first_last"] = (mem[0], mem[1], mem[-1])
    out["mem_growth_after_first_cycle"] = mem[-1] - mem[1]
    out["flashconv_alive_after_uninstall"] = sum(r() is not None for r in refs)
    sd1, md1 = sd_snapshot(m)
    out["after_sd_identical"] = list(sd1) == list(sd0) and all(torch.equal(sd1[k], sd0[k]) for k in sd0)
    out["after_md_identical"] = md1 == md0
    out["hooks_before_after"] = (h0, hooks_count(m))
    out["conv_tp_identity_restored"] = [id(b.conv_tp) for b in m.interactions] == convs0
    out["conv_fusion_left"] = [hasattr(b, "conv_fusion") for b in m.interactions]
    out["state_attr_left"] = hasattr(m, FA._STATE_ATTR)
    out["after_parity"] = cmp(ref, run(c, at), len(at[0]))
    try:
        torch.save(m, io.BytesIO()); out["torch_save_after_uninstall"] = "ok"
    except Exception as e:  # noqa
        out["torch_save_after_uninstall"] = f"{type(e).__name__}: {e}"
    ok &= out["after_sd_identical"] and out["after_md_identical"] and out["hooks_before_after"][1] == h0 \
        and out["conv_tp_identity_restored"] and not any(out["conv_fusion_left"]) and not out["state_attr_left"] \
        and out["mem_growth_after_first_cycle"] == 0 and out["after_parity"]["max_dF"] < 1e-13

    # partial install failure: FlashConv construction fails on layer 1 after layer 0 was built
    orig_cg = FA.cg_report
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure in layer 1")
        return orig_cg(*a, **k)
    FA.cg_report = bad
    try:
        FA.install(m)
        out["partial_install"] = "NO ERROR"
    except RuntimeError as e:
        out["partial_install"] = f"raised: {e}"
    finally:
        FA.cg_report = orig_cg
    out["partial_unchanged"] = ([id(b.conv_tp) for b in m.interactions] == convs0
                                and not any(hasattr(b, "conv_fusion") for b in m.interactions)
                                and not hasattr(m, FA._STATE_ATTR) and not FA.is_installed(m))
    ok &= out["partial_unchanged"]

    # hybrid: uninstall both -> stock state_dict
    ch = get("hybrid")
    mh = ch.models[0]
    FA.uninstall(mh); HB.uninstall_cueq_sc(mh)
    sdh, mdh = sd_snapshot(mh)
    out["hybrid_uninstalled_sd_equals_stock"] = list(sdh) == list(sd0) and all(torch.equal(sdh[k], sd0[k]) for k in sd0)
    out["hybrid_uninstalled_md_equals_stock"] = mdh == md0
    out["hybrid_uninstalled_hooks"] = hooks_count(mh)
    del CALCS[("hybrid", torch.float64)]  # no longer hybrid
    out["_ok"] = bool(ok)
    return out


def t_build_errors():
    out = {}
    for name, kw in [("bad_conv", dict(conv="x")), ("bad_path", dict(model_path="/nonexistent.model")),
                     ("bad_device_cuda7", dict(device="cuda:7")), ("dtype_f16", dict(dtype=torch.float16))]:
        try:
            HB.build_calc(**kw)
            out[name] = dict(err=None)
        except Exception as e:  # noqa
            out[name] = dict(err=f"{type(e).__name__}: {str(e)[:150]}")
        out[name]["default_after"] = str(torch.get_default_dtype())
        torch.set_default_dtype(DEF)
    # FA.install failure inside build_calc (after MACECalculator construction)
    orig = FA.install

    def bad(*a, **k):
        raise FA.AdapterError("injected")
    FA.install = bad
    try:
        HB.build_calc(conv="flashtp", sc="cueq")
        out["inject_install_fail"] = dict(err=None)
    except Exception as e:  # noqa
        out["inject_install_fail"] = dict(err=f"{type(e).__name__}: {e}")
    finally:
        FA.install = orig
    out["inject_install_fail"]["default_after"] = str(torch.get_default_dtype())
    out["_ok"] = all(v["default_after"] == str(DEF) for v in out.values())
    torch.set_default_dtype(DEF)
    return out


def t_repeat_build():
    out = {}
    gc.collect(); torch.cuda.synchronize()
    mem = []
    Es = []
    for i in range(5):
        c, _ = HB.build_calc(M, conv="flashtp", sc="cueq")
        E, F, _, _ = run(c, [WBOX_S])
        Es.append((E[0], F))
        del c, _
        gc.collect(); torch.cuda.synchronize()
        mem.append(torch.cuda.memory_allocated())
    out["mem_after_each_build"] = mem
    out["max_dE_across_builds"] = float(max(abs(e[0] - Es[0][0]) for e in Es))
    out["max_dF_across_builds"] = float(max(np.abs(e[1] - Es[0][1]).max() for e in Es))
    out["default_after"] = str(torch.get_default_dtype())
    return out


def t_run_to_run():
    out = {}
    for v in ("e3nn", "flash"):
        c = get(v)
        rs = [run(c, [WBOX], stress=True) for _ in range(20)]
        out[v] = dict(max_dE=float(max(abs(r[0][0] - rs[0][0][0]) for r in rs)),
                      max_dF=float(max(np.abs(r[1] - rs[0][1]).max() for r in rs)),
                      max_dS=float(max(np.abs(r[2] - rs[0][2]).max() for r in rs)),
                      n_distinct_E=len({float(r[0][0]) for r in rs}))
    c, _ = HB.build_calc(M, conv="flashtp", sc="cueq")
    rs = [run(c, [WBOX], stress=True) for _ in range(20)]
    out["hybrid"] = dict(max_dE=float(max(abs(r[0][0] - rs[0][0][0]) for r in rs)),
                         max_dF=float(max(np.abs(r[1] - rs[0][1]).max() for r in rs)),
                         n_distinct_E=len({float(r[0][0]) for r in rs}))
    return out


def t_input_checks():
    out = {}
    c = get("flash")
    fc = c.models[0].interactions[0].conv_tp
    n, e = 10, 40
    dev = fc.device
    nf = torch.randn(n, fc.in1_dim, dtype=torch.float64, device=dev)
    ea = torch.randn(e, fc.in2_dim, dtype=torch.float64, device=dev)
    w = torch.randn(e, fc.weight_numel, dtype=torch.float64, device=dev)
    ei = torch.randint(0, n, (2, e), device=dev)

    def tryit(name, *args):
        try:
            fc(*args); out[name] = "accepted"
        except FA.AdapterError as ex:
            out[name] = f"AdapterError: {str(ex)[:120]}"
        except Exception as ex:  # noqa
            out[name] = f"{type(ex).__name__}: {str(ex)[:120]}"
    bad = ei.clone(); bad[1, 3] = n
    tryit("index_eq_n", nf, ea, w, bad)
    bad = ei.clone(); bad[0, 0] = -1
    tryit("index_neg", nf, ea, w, bad)
    tryit("fp32_inputs", nf.float(), ea.float(), w.float(), ei)
    tryit("cpu_inputs", nf.cpu(), ea.cpu(), w.cpu(), ei.cpu())
    tryit("int32_index", nf, ea, w, ei.int())
    tryit("noncontig_nf", torch.randn(fc.in1_dim, n, dtype=torch.float64, device=dev).t(), ea, w, ei)
    tryit("ok", nf, ea, w, ei)
    out["_ok"] = all(out[k].startswith("AdapterError") for k in ("index_eq_n", "index_neg", "fp32_inputs", "cpu_inputs")) \
        and out["int32_index"] == "accepted" and out["ok"] == "accepted"
    return out


def t_index_overhead():
    out = {}
    c, _ = HB.build_calc(M, conv="flashtp", sc="e3nn")
    model = c.models[0]
    fcs = [b.conv_tp for b in model.interactions]
    for name, at in [("waterbox648", WBOX), ("ethanol", MOLS["CH3CH2OH"])]:
        b = make_batch(c, [at]).to_dict()

        def once():
            bb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch_tools.default_dtype(c.default_dtype):
                o = model(bb, compute_stress=False, training=False)
            o["forces"].sum().item()
            torch.cuda.synchronize()
            return time.perf_counter() - t
        for f in fcs: f.check_indices = True
        for _ in range(8): once()
        T = {True: [], False: []}
        for i in range(40):
            for flag in ((True, False) if i % 2 else (False, True)):
                for f in fcs: f.check_indices = flag
                T[flag].append(once())
        mt, mf = float(np.median(T[True])), float(np.median(T[False]))
        out[name] = dict(median_ms_check=mt * 1e3, median_ms_nocheck=mf * 1e3, overhead_ms=(mt - mf) * 1e3,
                         overhead_pct=100 * (mt - mf) / mf, n=40)
    for f in fcs: f.check_indices = True
    return out


TESTS = [("parity_fp64", t_parity_fp64), ("cg_tables", t_cg_tables), ("fp32", t_fp32),
         ("graph_changes", t_graph_changes), ("zero_edge", t_zero_edge), ("input_checks", t_input_checks),
         ("run_to_run", t_run_to_run), ("build_errors", t_build_errors), ("repeat_build", t_repeat_build),
         ("index_overhead", t_index_overhead), ("cleanup", t_cleanup)]
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for nm, fn in TESTS:
    if only is None or nm in only:
        record(nm, fn)
RES["default_at_end"] = str(torch.get_default_dtype())
save()
print("FAILURES:", FAIL)
