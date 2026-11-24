"""validate6 checks 1-3: recorded-kernel sanity, long changing-coordinate run + forced recaptures, edge cases.
usage: python -I v_correct.py OUT.json"""
import os, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import *  # noqa
from torch.profiler import profile, ProfilerActivity
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

out_path = sys.argv[1]
res = header(); res["failures"] = []
calcs, builders, models = build_all()
r_max = float(calcs["e3nn"].r_max); res["r_max"] = r_max
TOL_E, TOL_F = 1e-10, 1e-11   # pass thresholds: |dE|/atom (eV), max|dF| (eV/A) vs eager e3nn (fp64)


def save():
    res["gpu_end"] = gpuinfo(); json.dump(res, open(out_path, "w"), indent=1)


def ok(c):
    return c["finite"] and c["dE_per_atom"] < TOL_E and c["max_dF"] < TOL_F


def section(name, fn):
    print(f"##### {name}", flush=True)
    try:
        res[name] = fn()
    except Exception as ex:
        res[name] = dict(error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc())
        res["failures"].append(f"{name}: {type(ex).__name__}: {str(ex)[:300]}"); print(traceback.format_exc(), flush=True)
    save()


def gstep(G, bd):
    Eg, Fg = G.step(bd); return Eg.clone(), Fg.clone()


# ---------------------------------------------------------------------------------------------- 1. kernel sanity
FLASH_PAT = ("sptp",)
CUEQ_PAT = ("segmented", "cuequivariance", "uniform_1d", "tensor_product", "kernelcatalog", "fused_tp", "symmetric")


def kprofile(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    ev = [e for e in p.events() if str(e.device_type).endswith("CUDA")]
    names = {}
    t0 = min(e.time_range.start for e in ev) if ev else 0; t1 = max(e.time_range.end for e in ev) if ev else 0
    fl = cu = memcpy = 0; fl_t = cu_t = tot_t = 0.0; fl_names, cu_names = set(), set()
    for e in ev:
        d = e.time_range.end - e.time_range.start
        nm = e.name; low = nm.lower(); names[nm] = names.get(nm, 0) + 1; tot_t += d
        if "memcpy" in low or "memset" in low:
            memcpy += 1
        if any(s in low for s in FLASH_PAT):
            fl += 1; fl_t += d; fl_names.add(nm[:120])
        elif any(s in low for s in CUEQ_PAT):
            cu += 1; cu_t += d; cu_names.add(nm[:120])
    return dict(n_gpu_events=len(ev), n_memcpy_memset=memcpy, n_flash=fl, flash_us=fl_t, flash_names=sorted(fl_names),
                n_cueq=cu, cueq_us=cu_t, cueq_names=sorted(cu_names), sum_gpu_us=tot_t, span_us=t1 - t0,
                top=sorted(names.items(), key=lambda x: -x[1])[:12])


def event_time_ms(fn, n=30):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def check1():
    out = {}
    for sname in ("aspirin", "water648"):
        at = load(sname)
        for bk in ("hybrid", "flash"):
            bd = builders[bk].build(at)
            G = GraphedMACE(models[bk], bd, round_cap(bd["edge_index"].shape[1]), r_max)
            G.step(bd)
            pr = kprofile(G.replay)
            pe = kprofile(lambda: eager(models[bk], bd))
            rt = event_time_ms(G.replay)
            pr["replay_event_ms"] = rt
            pr["flash_plus_cueq_le_replay"] = (pr["flash_us"] + pr["cueq_us"]) / 1e3 <= rt
            pr["sum_gpu_le_replay"] = pr["sum_gpu_us"] / 1e3 <= rt * 1.02
            exp_flash = 2 * 2  # 2 layers x (fwd + bwd) for a force call
            pr["flash_kernels_present"] = pr["n_flash"] >= exp_flash
            pr["cueq_kernels_present"] = (pr["n_cueq"] > 0) if bk == "hybrid" else True
            out[f"{sname}/{bk}"] = dict(replay=pr, eager=pe,
                                        pass_=pr["flash_kernels_present"] and pr["cueq_kernels_present"] and pr["sum_gpu_le_replay"]
                                        and pr["n_flash"] == pe["n_flash"] and pr["n_cueq"] == pe["n_cueq"])
            print(sname, bk, {k: pr[k] for k in ("n_gpu_events", "n_flash", "flash_us", "n_cueq", "cueq_us", "sum_gpu_us", "span_us", "replay_event_ms")},
                  "eager", {k: pe[k] for k in ("n_gpu_events", "n_flash", "n_cueq", "sum_gpu_us")}, flush=True)
            del G; torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------- 2. long run
def check2():
    out = {}
    at = load("water192"); n = len(at)
    at.calc = calcs["e3nn"]
    MaxwellBoltzmannDistribution(at, temperature_K=300, rng=np.random.default_rng(1234))
    frames = []
    dyn = VelocityVerlet(at, timestep=0.5 * units.fs)
    def grab():
        a = at.copy(); a.calc = None; frames.append((dyn.nsteps, a, float(at.get_potential_energy()), at.get_forces().copy()))
    dyn.attach(grab, interval=5)
    t0 = time.perf_counter(); dyn.run(200); out["md_wall_s"] = time.perf_counter() - t0
    out["n_frames"] = len(frames)
    bd0 = builders["e3nn"].build(frames[0][1]); E0 = bd0["edge_index"].shape[1]
    Gs = {bk: GraphedMACE(models[bk], builders[bk].build(frames[0][1]), round_cap(E0), r_max) for bk in ("hybrid", "flash")}
    rows = []
    for step, a, Estock, Fstock in frames:
        bd = builders["e3nn"].build(a); Er, Fr = eager(models["e3nn"], bd)
        row = dict(step=step, E=int(bd["edge_index"].shape[1]),
                   proto_vs_stock_calc=cmp(Er, torch.tensor(Fstock, device=Fr.device), torch.tensor(Estock, device=Er.device, dtype=Er.dtype), Fr, n))
        for bk, G in Gs.items():
            Eg, Fg = gstep(G, builders[bk].build(a)); row[bk] = cmp(Eg, Fg, Er, Fr, n)
            row[bk]["ok"] = ok(row[bk])
        rows.append(row)
    out["frames"] = rows
    for bk in Gs:
        out[f"{bk}_max_dE_per_atom"] = max(r[bk]["dE_per_atom"] for r in rows)
        out[f"{bk}_max_dF"] = max(r[bk]["max_dF"] for r in rows)
        out[f"{bk}_max_rms_dF"] = max(r[bk]["rms_dF"] for r in rows)
        out[f"{bk}_recaptures"] = Gs[bk].recaptures
        out[f"{bk}_E_cap"] = Gs[bk].E_cap
    out["E_range"] = [min(r["E"] for r in rows), max(r["E"] for r in rows)]
    out["pass_"] = all(r[bk]["ok"] for r in rows for bk in Gs)
    print({k: v for k, v in out.items() if k != "frames"}, flush=True)
    del Gs; torch.cuda.empty_cache()
    return out


def check2b():
    out = {}
    n = 192
    # ---- forced recaptures: compress water192 about its centroid (edge count grows), then expand back
    at0 = load("water192"); c = at0.positions.mean(0)
    scales = list(np.linspace(1.0, 0.78, 12)) + [1.0, 0.9, 0.78]
    seq = []
    for s in scales:
        a = at0.copy(); a.positions = c + (at0.positions - c) * s; seq.append((float(s), a))
    rc = {}
    for bk in ("hybrid", "flash"):
        bdA = builders[bk].build(seq[0][1]); EA = bdA["edge_index"].shape[1]
        G = GraphedMACE(models[bk], bdA, int(np.ceil(EA / 64)) * 64, r_max)  # deliberately tight capacity
        torch.cuda.synchronize(); m_start = torch.cuda.memory_allocated()
        rr = []
        for s, a in seq:
            nrc = len(G.recaptures)
            bd = builders[bk].build(a); Eg, Fg = gstep(G, bd)
            Er, Fr = eager(models["e3nn"], builders["e3nn"].build(a))
            r = dict(scale=s, E=int(bd["edge_index"].shape[1]), E_cap=G.E_cap, recaptured=len(G.recaptures) > nrc, **cmp(Eg, Fg, Er, Fr, n))
            Ee, Fe = eager(models[bk], bd)
            r["eager_same_vs_e3nn"] = cmp(Ee, Fe, Er, Fr, n); r["graphed_vs_eager_same"] = cmp(Eg, Fg, Ee, Fe, n)
            r["max_absF"] = float(Fr.abs().max()); r["rel_dF"] = r["max_dF"] / r["max_absF"]
            r["ok"] = r["finite"] and r["graphed_vs_eager_same"]["max_dF"] < 1e-11 and r["rel_dF"] < 1e-14
            # replay again right after (recaptured graph reused, no copy-in)
            Eg2, Fg2 = G.replay(); r["rereplay_max_dF"] = float((Fg2 - Fg).abs().max())
            rr.append(r)
        torch.cuda.synchronize()
        rc[bk] = dict(rows=rr, recaptures=G.recaptures, n_recaptures=len(G.recaptures),
                      mem_growth_MiB=(torch.cuda.memory_allocated() - m_start) / 2**20,
                      pass_=all(r["ok"] for r in rr) and len(G.recaptures) >= 2)
        print(bk, "recaptures", G.recaptures, "all ok", rc[bk]["pass_"], [(r["E"], r["E_cap"], r["recaptured"], "%.1e" % r["max_dF"], "%.1e" % r["eager_same_vs_e3nn"]["max_dF"], "%.1e" % r["graphed_vs_eager_same"]["max_dF"], "%.0f" % r["max_absF"]) for r in rr], flush=True)
        del G; torch.cuda.empty_cache()
    out["recapture"] = rc
    return out


# ---------------------------------------------------------------------------------------------- 3. edge cases
def trunc(bd, E):
    b = dict(bd)
    b["edge_index"] = bd["edge_index"][:, :E].contiguous(); b["shifts"] = bd["shifts"][:E].contiguous()
    b["unit_shifts"] = bd["unit_shifts"][:E].contiguous(); return b


def perturb(at, sig, seed):
    a = at.copy(); a.positions = a.positions + np.random.default_rng(seed).normal(0, sig, a.positions.shape); return a


def check3():
    out = {}
    # zero-edge
    for sname in ("single_C", "far_pair"):
        at = load(sname); n = len(at)
        Er, Fr = eager(models["e3nn"], builders["e3nn"].build(at))
        for bk in ("hybrid", "flash"):
            bd = builders[bk].build(at)
            G = GraphedMACE(models[bk], bd, round_cap(bd["edge_index"].shape[1]), r_max)
            Eg, Fg = gstep(G, bd)
            r = cmp(Eg, Fg, Er, Fr, n); r["E_edges"] = int(bd["edge_index"].shape[1]); r["F_exact_zero"] = bool((Fg == 0).all())
            r["E_exact"] = bool((Eg == Er).all()); r["ok"] = ok(r) and r["F_exact_zero"]
            out[f"zero_edge/{sname}/{bk}"] = r; del G
    # E exactly at capacity, and capacity+1 (truncated edge list = valid graph, identical for eager & graphed)
    at = load("aspirin"); n = len(at)
    for bk in ("hybrid", "flash"):
        bd = builders[bk].build(at); E = bd["edge_index"].shape[1]
        bt = trunc(bd, E - 1)
        Ert, Frt = eager(models["e3nn"], trunc(builders["e3nn"].build(at), E - 1))
        Er, Fr = eager(models["e3nn"], builders["e3nn"].build(at))
        assert torch.equal(builders["e3nn"].build(at)["edge_index"], bd["edge_index"])
        G = GraphedMACE(models[bk], bt, E - 1, r_max)
        r1 = cmp(*gstep(G, bt), Ert, Frt, n); r1["E"] = E - 1; r1["E_cap"] = G.E_cap; r1["ok"] = ok(r1)
        r2 = cmp(*gstep(G, bd), Er, Fr, n); r2["E"] = E; r2["recaptures"] = list(G.recaptures); r2["E_cap"] = G.E_cap
        r2["ok"] = ok(r2) and len(G.recaptures) == 1
        r3 = cmp(*gstep(G, bt), Ert, Frt, n); r3["ok"] = ok(r3)
        out[f"cap_exact/{bk}"] = r1; out[f"cap_plus1/{bk}"] = r2; out[f"cap_back_to_minus1/{bk}"] = r3
        del G
    # periodic box
    at = load("wbox648"); n = len(at)
    for bk in ("hybrid", "flash"):
        bd = builders[bk].build(at); G = GraphedMACE(models[bk], bd, round_cap(bd["edge_index"].shape[1]), r_max)
        rows = []
        for k in range(4):
            a = at if k == 0 else perturb(at, 0.03, k)
            Er, Fr = eager(models["e3nn"], builders["e3nn"].build(a))
            r = cmp(*gstep(G, builders[bk].build(a)), Er, Fr, n); r["E"] = int(builders[bk].build(a)["edge_index"].shape[1]); r["ok"] = ok(r); rows.append(r)
        out[f"periodic_wbox648/{bk}"] = dict(rows=rows, ok=all(r["ok"] for r in rows), pbc=bool(any(at.pbc)))
        del G; torch.cuda.empty_cache()
    # multiple graphed models alive + stale buffers A,B,A
    sysn = ("aspirin", "water192")
    geo = {s: [load(s), perturb(load(s), 0.05, 7)] for s in sysn}
    refs = {(s, i): eager(models["e3nn"], builders["e3nn"].build(geo[s][i])) for s in sysn for i in range(2)}
    Gs = {(s, bk): GraphedMACE(models[bk], builders[bk].build(geo[s][0]), round_cap(builders[bk].build(geo[s][0])["edge_index"].shape[1]), r_max)
          for s in sysn for bk in ("hybrid", "flash")}
    order = [0, 1, 0, 1, 1, 0]
    rows = []
    for i in order:
        for (s, bk), G in Gs.items():
            r = cmp(*gstep(G, builders[bk].build(geo[s][i])), *refs[(s, i)], len(geo[s][i])); r.update(sys=s, bk=bk, geo="AB"[i]); r["ok"] = ok(r); rows.append(r)
    # outputs of one graph untouched by another's replay
    Ga, Gb = Gs[("aspirin", "hybrid")], Gs[("water192", "hybrid")]
    Ga.step(builders["hybrid"].build(geo["aspirin"][0])); Ea, Fa = Ga.out_E.clone(), Ga.out_F.clone()
    for _ in range(3):
        Gb.step(builders["hybrid"].build(geo["water192"][1]))
    torch.cuda.synchronize()
    iso = bool(torch.equal(Ea, Ga.out_E) and torch.equal(Fa, Ga.out_F))
    out["multi_model_ABA"] = dict(rows=rows, ok=all(r["ok"] for r in rows), outputs_isolated=iso, n_graphs_alive=len(Gs))
    del Gs, Ga, Gb; torch.cuda.empty_cache()
    # API hazard: same N, different species -> node_attrs not refreshed by step()
    at = load("aspirin"); ax = at.copy(); ax.numbers = at.numbers[::-1].copy()
    G = GraphedMACE(models["hybrid"], builders["hybrid"].build(at), 512, r_max)
    Er, Fr = eager(models["e3nn"], builders["e3nn"].build(ax))
    r = cmp(*gstep(G, builders["hybrid"].build(ax)), Er, Fr, len(ax)); r["silently_wrong"] = not ok(r)
    out["hazard_same_N_diff_species"] = r
    try:
        G.step(builders["hybrid"].build(load("water192"))); out["hazard_diff_N"] = "no error (BAD)"
    except Exception as ex:
        out["hazard_diff_N"] = f"raised {type(ex).__name__}: {str(ex)[:200]}"
    # API hazard: returned tensors are views overwritten by next step
    out["hazard_diff_N_state_after"] = dict(N_attr=G.N, static_positions_rows=int(G.static["positions"].shape[0]), E_cap=G.E_cap, recaptures=list(G.recaptures))
    if G.recaptures:  # step() silently recaptured for the other system: is that result correct for water192?
        w = load("water192"); Erw, Frw = eager(models["e3nn"], builders["e3nn"].build(w))
        r = cmp(*gstep(G, builders["hybrid"].build(w)), Erw, Frw, len(w)); out["hazard_diff_N_result_vs_eager_water192"] = r
    try:
        G.step(builders["hybrid"].build(at)); out["hazard_back_to_original"] = "ok"
    except Exception as ex:
        out["hazard_back_to_original"] = f"raised {type(ex).__name__}: {str(ex)[:200]}"
    G = GraphedMACE(models["hybrid"], builders["hybrid"].build(at), 512, r_max)
    E1, F1 = G.step(builders["hybrid"].build(at)); keep = F1.clone(); G.step(builders["hybrid"].build(perturb(at, 0.05, 3)))
    out["hazard_outputs_are_views"] = bool(not torch.equal(F1, keep))
    del G
    out["pass_"] = all(v.get("ok", True) for k, v in out.items() if isinstance(v, dict) and not k.startswith("hazard"))
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v, flush=True)
    return out


SEL = sys.argv[2].split(",") if len(sys.argv) > 2 else ["check1_kernels", "check2_long_run", "check2b_recapture", "check3_edge_cases"]
for nm, fn in (("check1_kernels", check1), ("check2_long_run", check2), ("check2b_recapture", check2b), ("check3_edge_cases", check3)):
    if nm in SEL:
        section(nm, fn)
save()
print("DONE failures:", res["failures"])
