"""Profile MACE-OFF23 medium model compute on a fixed prebuilt graph (fp64), backends e3nn/flash/hybrid (v2).
Usage: python -I prof6.py <out_json> <backends comma> <systems...>   env: NPROF=3 NTIME=30 WARM=10 TRACES=1
Phases: fwd = model(bd, compute_force=False); bwd = autograd.grad(E, positions) after an un-timed fwd;
fwd_bwd = model(bd, compute_force=True). check_indices=False for FlashTP."""
import os, sys, json, time, gzip, subprocess, bisect, collections
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/flash_v2")
import hybrid_v2 as HB   # first
sys.path.insert(0, f"{R}/scripts/workloads")
from systems2 import load_system
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity, record_function
from mace import data as mace_data
from mace.tools import torch_tools

out_path = sys.argv[1]; BACKENDS = sys.argv[2].split(","); NAMES = sys.argv[3:]
NPROF = int(os.environ.get("NPROF", 3)); NTIME = int(os.environ.get("NTIME", 30)); WARM = int(os.environ.get("WARM", 10))
TRACES = os.environ.get("TRACES", "1") == "1"
M = f"{R}/models/MACE-OFF23_medium.model"; RES = f"{R}/results/profile6"
CFG = dict(e3nn=("e3nn", "e3nn"), flash=("flashtp", "e3nn"), hybrid=("flashtp", "cueq"))
sync = torch.cuda.synchronize


def nvsmi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


out = dict(slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, nvidia_smi_start=nvsmi(), torch=torch.__version__,
           nprof=NPROF, ntime=NTIME, warm=WARM, check_indices=False, backends={}, graph_construction={}, argv=sys.argv)
def save(): json.dump(out, open(out_path, "w"), indent=1)


def make_batch_dict(calc, atoms):   # == scripts/workloads/bench_wl.py
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v): b[k] = v.to(dtype=torch.float64)
    return b.to_dict()


def graph_cpu(calc, atoms):
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    with torch_tools.default_dtype("float64"):
        cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
        return mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)


# ------------------------------------------------------------------ annotation hooks (stack with gap ranges)
class Annot:
    def __init__(self, model):
        self.model = model; self.stack = []; self.handles = []; self.enabled = True
        names = ["", "node_embedding", "spherical_harmonics", "radial_embedding", "scale_shift"]
        for i, _ in enumerate(model.interactions):
            names += [f"interactions.{i}"] + [f"interactions.{i}.{s}" for s in ("skip_tp", "linear_up", "conv_tp_weights", "conv_tp", "linear")]
        for i, _ in enumerate(model.products):
            names += [f"products.{i}", f"products.{i}.symmetric_contractions", f"products.{i}.linear"]
        for i, _ in enumerate(model.readouts): names.append(f"readouts.{i}")
        mods = dict(model.named_modules()); self.names = []
        for n in names:
            m = mods[n]; label = n or "model"; self.names.append(label)
            self.handles.append(m.register_forward_pre_hook(lambda mod, a, label=label: self.pre(label)))
            self.handles.append(m.register_forward_hook(lambda mod, a, o, label=label: self.post(label, o)))
        self.shapes = None

    def cur(self):
        if not self.stack: return "none"
        e = self.stack[-1]
        return e[2][0] if e[2] is not None else e[0]

    def pre(self, label):
        if not self.enabled: return
        if self.stack and self.stack[-1][2] is not None:
            self.stack[-1][2][1].__exit__(None, None, None); self.stack[-1][2] = None
        rf = record_function("A:" + label); rf.__enter__(); self.stack.append([label, rf, None])

    def post(self, label, o):
        if not self.enabled: return
        e = self.stack.pop(); assert e[0] == label, (e[0], label)
        if e[2] is not None: e[2][1].__exit__(None, None, None)
        e[1].__exit__(None, None, None)
        if self.shapes is not None:
            os_ = o if isinstance(o, (tuple, list)) else (o,)
            self.shapes.setdefault(label, [dict(shape=list(t.shape), dtype=str(t.dtype).replace("torch.", ""), MB=t.numel() * t.element_size() / 1e6)
                                           for t in os_ if torch.is_tensor(t)])
        if self.stack:
            g = f"{self.stack[-1][0]}/after:{label.split('.')[-1]}"
            rf = record_function("A:" + g); rf.__enter__(); self.stack[-1][2] = [g, rf]


# ------------------------------------------------------------------ trace analysis
LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaLaunchCooperativeKernel")
SYNCN = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cuStreamSynchronize", "cuCtxSynchronize")


def union_len(iv):
    iv = sorted(iv); tot = 0; ce = -1e30; cs = None
    for s, e in iv:
        if s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot


def op_group(label):
    """Map innermost annotation label -> MACE op group."""
    import re
    m = re.match(r"interactions\.(\d+)(?:\.(\w+))?(?:/after:(\w+))?$", label)
    if m:
        i, sub, aft = m.groups()
        if sub in ("conv_tp",) or aft in ("conv_tp_weights", "conv_tp"): return f"L{i}.conv"
        if sub == "linear" or aft == "linear": return f"L{i}.linear(+/avg,reshape)"
        if sub: return f"L{i}.{sub}"
        if aft == "skip_tp": return f"L{i}.skip_tp"
        if aft == "linear_up": return f"L{i}.linear_up"
        return f"L{i}.interaction_other"
    m = re.match(r"products\.(\d+)(?:\.(\w+))?(?:/after:(\w+))?$", label)
    if m:
        i, sub, aft = m.groups()
        if sub == "symmetric_contractions" or aft == "symmetric_contractions": return f"P{i}.symmetric_contractions"
        if sub == "linear" or aft == "linear": return f"P{i}.linear(+sc)"
        return f"P{i}.product_other"
    if label.startswith("readouts"): return "readouts"
    if label.startswith("model/after:"):
        a = label.split(":")[1]
        if a in ("readouts", "0", "1") or a.startswith("readout"): return "energy_sum/outputs"
        if a == "scale_shift": return "energy_sum/outputs"
        if a == "radial_embedding": return "pre_interaction_glue"
        return f"glue_after_{a}"
    if label == "model": return "prepare_graph/e0"
    return label


def analyze(trace_path):
    ev = json.load(open(trace_path, 'r'))["traceEvents"]
    ev = [e for e in ev if e.get("ph") == "X"]
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    ua = [e for e in ev if e.get("cat") == "user_annotation"]
    cpuop = [e for e in ev if e.get("cat") == "cpu_op"]
    k_by_corr = collections.defaultdict(list)
    for k in kern: k_by_corr[k["args"].get("correlation")].append(k)
    # intervals per thread for innermost lookup
    def by_tid(lst):
        d = collections.defaultdict(list)
        for e in lst: d[e["tid"]].append(e)
        for v in d.values(): v.sort(key=lambda e: e["ts"])
        return d
    ua_t = by_tid([e for e in ua if e["name"].startswith("A:")]); op_t = by_tid(cpuop)
    def innermost(d, tid, ts, pred=lambda e: True):
        best = None
        for e in d.get(tid, []):
            if e["ts"] > ts: break
            if e["ts"] <= ts <= e["ts"] + e["dur"] and pred(e):
                if best is None or e["dur"] <= best["dur"]: best = e
        return best
    # forward seq -> label (innermost annotation enclosing the fwd op)
    seq2label = {}
    for e in cpuop:
        s = e.get("args", {}).get("Sequence number")
        if s is None or e["name"].startswith("autograd::engine"): continue
        a = innermost(ua_t, e["tid"], e["ts"])
        if a is not None and s not in seq2label: seq2label[s] = a["name"][2:]
    phases = [e for e in ua if e["name"].startswith("PHASE:")]
    res = collections.defaultdict(list)
    for ph in phases:
        pname = ph["name"][6:]; t0, t1 = ph["ts"], ph["ts"] + ph["dur"]
        rts = [r for r in rt if t0 <= r["ts"] <= t1]
        launches = [r for r in rts if r["name"] in LAUNCH or "LaunchKernel" in r["name"]]
        ks = [k for r in rts for k in k_by_corr.get(r["args"].get("correlation"), [])]
        kk = [k for k in ks if k["cat"] == "kernel"]
        ivs = [(k["ts"], k["ts"] + k["dur"]) for k in ks]
        # wall = phase range (ends after explicit synchronize)
        syncs = []
        for r in rts:
            nm = r["name"]
            is_memcpy = "Memcpy" in nm or "memcpy" in nm.lower()
            d2h = any("DtoH" in k["name"] for k in k_by_corr.get(r["args"].get("correlation"), []))
            if nm in SYNCN or (is_memcpy and d2h) or (nm.startswith("cudaMemcpy") and "Async" not in nm):
                endsync = innermost(ua_t | {}, r["tid"], r["ts"]) if False else None
                op = innermost(op_t, r["tid"], r["ts"]); a = innermost(ua_t, r["tid"], r["ts"])
                # explicit end-of-phase synchronize is called inside PHASE but outside model annotations
                opn = op["name"] if op else None
                syncs.append(dict(api=nm, dur_us=r["dur"], cpu_op=opn, annot=a["name"][2:] if a else None,
                                  explicit_phase_end=(opn is None and a is None and nm == "cudaDeviceSynchronize")))
        # per-op attribution
        per = collections.defaultdict(lambda: dict(gpu_us=0.0, launches=0, cpu_us=0.0, bwd_gpu_us=0.0, bwd_launches=0, bwd_cpu_us=0.0))
        for r in launches + [r for r in rts if r["name"].startswith("cudaMemcpy") or r["name"].startswith("cudaMemset")]:
            evf = innermost(op_t, r["tid"], r["ts"], lambda e: e["name"].startswith("autograd::engine::evaluate_function"))
            kl = k_by_corr.get(r["args"].get("correlation"), [])
            if evf is not None:
                s = evf["args"].get("Sequence number"); lab = seq2label.get(s, "bwd_unmapped:" + evf["name"].split(": ")[-1])
                g = per[op_group(lab)]; g["bwd_gpu_us"] += sum(k["dur"] for k in kl); g["bwd_launches"] += 1 if r in launches else 0
            else:
                a = innermost(ua_t, r["tid"], r["ts"]); lab = a["name"][2:] if a else "outside_model"
                g = per[op_group(lab)]; g["gpu_us"] += sum(k["dur"] for k in kl); g["launches"] += 1 if r in launches else 0
        # cpu time per op (fwd): exclusive time of annotation ranges inside phase (innermost labels are leaves or gap-split)
        for tid, lst in ua_t.items():
            inph = [e for e in lst if t0 <= e["ts"] <= t1]
            for e in inph:
                child = sum(c["dur"] for c in inph if c is not e and e["ts"] <= c["ts"] and c["ts"] + c["dur"] <= e["ts"] + e["dur"]
                            and not any(x is not e and x is not c and e["ts"] <= x["ts"] <= c["ts"] and c["ts"] + c["dur"] <= x["ts"] + x["dur"] and x["dur"] < e["dur"] for x in inph))
                per[op_group(e["name"][2:])]["cpu_us"] += e["dur"] - child
        cpuops_by = collections.defaultdict(collections.Counter)
        for e in cpuop:
            if t0 <= e["ts"] <= t1 and not e["name"].startswith("autograd::engine"):
                a = innermost(ua_t, e["tid"], e["ts"])
                if a is not None and innermost(op_t, e["tid"], e["ts"], lambda x: x is not e and x["dur"] >= e["dur"] and innermost(ua_t, e["tid"], x["ts"]) is a) is None:
                    cpuops_by[op_group(a["name"][2:])][e["name"]] += e["dur"]   # top-level cpu ops directly under the annotation
        for g, c in cpuops_by.items(): per[g]["top_cpu_ops_us"] = [(n, round(v, 1)) for n, v in c.most_common(4)]
        bwd_threads = set()
        for tid, lst in op_t.items():
            for e in lst:
                if t0 <= e["ts"] <= t1 and e["name"].startswith("autograd::engine::evaluate_function"):
                    s = e["args"].get("Sequence number"); lab = seq2label.get(s, "bwd_unmapped:" + e["name"].split(": ")[-1])
                    per[op_group(lab)]["bwd_cpu_us"] += e["dur"]; bwd_threads.add(tid)
        topk = collections.defaultdict(lambda: [0.0, 0])
        for k in kk: topk[k["name"]][0] += k["dur"]; topk[k["name"]][1] += 1
        res[pname].append(dict(
            wall_us=ph["dur"], kernel_sum_us=sum(k["dur"] for k in kk), busy_union_us=union_len(ivs),
            busy_union_kernels_only_us=union_len([(k["ts"], k["ts"] + k["dur"]) for k in kk]),
            gpu_first_last_span_us=(max(e for _, e in ivs) - min(s for s, _ in ivs)) if ivs else 0,
            n_kernels=len(kk), n_launch_api=len(launches), launch_api_us=sum(r["dur"] for r in launches),
            n_memcpy=sum(1 for k in ks if k["cat"] == "gpu_memcpy"), n_memset=sum(1 for k in ks if k["cat"] == "gpu_memset"),
            memcpy_kinds=dict(collections.Counter(k["name"] for k in ks if k["cat"] == "gpu_memcpy")),
            runtime_api_us=sum(r["dur"] for r in rts), runtime_api_counts=dict(collections.Counter(r["name"] for r in rts)),
            syncs=syncs, per_op={k: dict(v) for k, v in per.items()},
            top_kernels=sorted([(n, v[0], v[1]) for n, v in topk.items()], key=lambda x: -x[1])[:15]))
    return res


def agg(lst):
    """median over profiled iterations; per_op / top kernels from the median-wall iteration."""
    if not lst: return None
    keys = [k for k, v in lst[0].items() if isinstance(v, (int, float))]
    d = {k: float(np.median([x[k] for x in lst])) for k in keys}
    mid = sorted(lst, key=lambda x: x["wall_us"])[len(lst) // 2]
    d.update(idle_frac=1 - d["busy_union_us"] / d["wall_us"], per_op=mid["per_op"], top_kernels=mid["top_kernels"],
             syncs=mid["syncs"], runtime_api_counts=mid["runtime_api_counts"], memcpy_kinds=mid["memcpy_kinds"])
    return d


# ------------------------------------------------------------------ main
for be in BACKENDS:
    conv, sc = CFG[be]
    calc, info = HB.build_calc(M, conv=conv, sc=sc, check_indices=False)
    model = calc.models[0]
    ob = out["backends"][be] = dict(variant=info["variant"], backend_map=info["backend_map"], systems={})
    def fwd(bd):
        with torch_tools.default_dtype("float64"): return model(bd, compute_force=False, training=False)
    def fwd_bwd(bd):
        with torch_tools.default_dtype("float64"): return model(bd, compute_force=True, training=False)
    def bwd(o, bd):
        return torch.autograd.grad([o["energy"]], [bd["positions"]], grad_outputs=[torch.ones_like(o["energy"])])
    for name in NAMES:
        atoms = load_system(name)
        bd = make_batch_dict(calc, atoms)
        d = ob["systems"][name] = dict(n_atoms=len(atoms), n_edges=int(bd["edge_index"].shape[1]))
        # graph construction (once per system; backend independent -> only for first backend)
        if be == BACKENDS[0]:
            for _ in range(5): make_batch_dict(calc, atoms); graph_cpu(calc, atoms)
            tg, tt = [], []
            for _ in range(20):
                sync(); t = time.perf_counter(); graph_cpu(calc, atoms); tg.append((time.perf_counter() - t) * 1e3)
                sync(); t = time.perf_counter(); make_batch_dict(calc, atoms); sync(); tt.append((time.perf_counter() - t) * 1e3)
            gtr = f"{RES}/tmp_graph_{name}.json"
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                with record_function("PHASE:graph"):
                    make_batch_dict(calc, atoms); sync()
            p.export_chrome_trace(gtr); ga = analyze(gtr)["graph"][0]; os.remove(gtr)
            gev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CPU]
            out["graph_construction"][name] = dict(graph_cpu_ms_median=float(np.median(tg)), graph_total_ms_median=float(np.median(tt)),
                h2d_tensor_part_ms=float(np.median(tt) - np.median(tg)), profiled_wall_us=ga["wall_us"], n_memcpy=ga["n_memcpy"],
                memcpy_kinds=ga["memcpy_kinds"], n_kernels=ga["n_kernels"], kernel_sum_us=ga["kernel_sum_us"],
                runtime_api_counts=ga["runtime_api_counts"], runtime_api_us=ga["runtime_api_us"],
                n_syncs=len(ga["syncs"]), syncs=ga["syncs"])
        # warmup
        for _ in range(WARM):
            fwd(bd); o = fwd(bd); bwd(o, bd); fwd_bwd(bd)
        sync()
        # un-profiled timing, no hooks: perf_counter (synced) + cuda events
        tm = {}
        for ph in ("fwd", "bwd", "fwd_bwd"):
            w, ge = [], []
            for _ in range(NTIME):
                o = fwd(bd) if ph == "bwd" else None
                sync(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t = time.perf_counter(); e0.record()
                if ph == "fwd": fwd(bd)
                elif ph == "bwd": bwd(o, bd)
                else: fwd_bwd(bd)
                e1.record(); sync(); w.append((time.perf_counter() - t) * 1e3); ge.append(e0.elapsed_time(e1))
            tm[ph] = dict(wall_ms_median=float(np.median(w)), wall_ms_p10=float(np.percentile(w, 10)), wall_ms_p90=float(np.percentile(w, 90)),
                          cuda_event_ms_median=float(np.median(ge)))
        d["timing_unprofiled"] = tm
        # intermediate shapes + saved-for-backward (one fwd pass with hooks, no profiler)
        A = Annot(model)
        if name in ("aspirin", "water648"):
            A.shapes = {}; saved = collections.defaultdict(lambda: dict(count=0, MB=0.0)); seen = set()
            def pack(t):
                key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape))
                g = op_group(A.cur()); s = saved[g]; s["count"] += 1
                if key not in seen and t.device.type == "cuda":
                    seen.add(key); s["MB"] += t.numel() * t.element_size() / 1e6
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                o = fwd(bd)
            del o; sync()
            d["intermediate_shapes"] = A.shapes; d["saved_for_backward"] = {k: dict(v) for k, v in saved.items()}
            d["saved_for_backward_total_MB"] = sum(v["MB"] for v in saved.values()); A.shapes = None
        for _ in range(3): fwd(bd); o = fwd(bd); bwd(o, bd); fwd_bwd(bd)
        sync()
        tr = f"{RES}/trace_{be}_{name}.json"
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as p:
            for it in range(NPROF):
                sync()
                with record_function("PHASE:fwd"):
                    fwd(bd); torch.cuda.synchronize()
                o = fwd(bd); sync()
                with record_function("PHASE:bwd"):
                    bwd(o, bd); torch.cuda.synchronize()
                del o
                with record_function("PHASE:fwd_bwd"):
                    fwd_bwd(bd); torch.cuda.synchronize()
        p.export_chrome_trace(tr)
        an = analyze(tr)
        d["profiled"] = {ph: agg(v) for ph, v in an.items()}
        if TRACES:
            with open(tr, "rb") as f, gzip.open(tr + ".gz", "wb", compresslevel=6) as g: g.write(f.read())
        os.remove(tr)
        for h in A.handles: h.remove()
        pr = d["profiled"]
        print("RES", be, name, d["n_atoms"], d["n_edges"], json.dumps({ph: dict(wall=round(tm[ph]["wall_ms_median"], 2),
              prof_wall=round(pr[ph]["wall_us"] / 1e3, 2), busy=round(pr[ph]["busy_union_us"] / 1e3, 2), idle=round(pr[ph]["idle_frac"], 2),
              nk=pr[ph]["n_kernels"], launch_ms=round(pr[ph]["launch_api_us"] / 1e3, 2), nsync=len(pr[ph]["syncs"])) for ph in tm}), flush=True)
        save()
    del calc, model; torch.cuda.empty_cache()
out["nvidia_smi_end"] = nvsmi(); save(); print("DONE", flush=True)
