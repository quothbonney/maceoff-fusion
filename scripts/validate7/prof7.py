"""Phase-7 profile of the GRAPHED path (cudagraph6 GraphedMACE, one graph per system), backends flash / hybrid.
usage: python -I prof7.py OUT.json backends(comma) systems(comma)
Per system/backend:
  * replay-only wall (60 synced reps) and replay with a tight edge cap (pad -> multiple of 64 only) to measure padding cost
  * torch.profiler (CUDA) on 1 replay -> per-kernel GPU time, busy union, span
  * eager annotated profile of the SAME padded static inputs (record_function module hooks, prof6 style) -> kernel sequence
    with MACE-op labels (backward kernels mapped through autograd sequence numbers); aligned 1:1 with the replay
    kernel sequence by order + name (checked)
  * per-op shapes (module inputs/outputs) -> minimum bytes moved, FLOP estimates for linears/radial MLP
"""
import os, sys, json, time, re, collections, math, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/cudagraph6")
import hybrid_gc as HB  # first
sys.path.insert(1, f"{R}/scripts/graph")
import numpy as np
import torch
import ase.io
from torch.profiler import profile, ProfilerActivity, record_function
from mace.tools import torch_tools
from proto_graph import ProtoBuilder
from graphed_model import GraphedMACE, round_cap

out_path = sys.argv[1]
BACK = sys.argv[2].split(",")
SYS = sys.argv[3].split(",")
TMP = os.path.dirname(out_path)
dev = torch.device("cuda")


def load(name):
    return ase.io.read(f"{R}/results/workloads/systems/{name}.extxyz")


def timeit(fn, warm=10, reps=60):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))


def union_len(iv):
    iv = sorted(iv); tot = 0; ce = -1e30; cs = None
    for s, e in iv:
        if s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot


# ---------------------------------------------------------------- annotations (from profile6, + input shapes)
class Annot:
    def __init__(self, model):
        self.stack = []; self.handles = []; self.io = {}
        names = ["", "node_embedding", "spherical_harmonics", "radial_embedding", "scale_shift"]
        for i, _ in enumerate(model.interactions):
            names += [f"interactions.{i}"] + [f"interactions.{i}.{s}" for s in ("skip_tp", "linear_up", "conv_tp_weights", "conv_tp", "linear")]
        for i, _ in enumerate(model.products):
            names += [f"products.{i}", f"products.{i}.symmetric_contractions", f"products.{i}.linear"]
        for i, _ in enumerate(model.readouts): names.append(f"readouts.{i}")
        mods = dict(model.named_modules())
        for n in names:
            if n not in mods: continue
            m = mods[n]; label = n or "model"
            self.handles.append(m.register_forward_pre_hook(lambda mod, a, label=label: self.pre(label, a)))
            self.handles.append(m.register_forward_hook(lambda mod, a, o, label=label: self.post(label, o)))

    def pre(self, label, a):
        if self.stack and self.stack[-1][2] is not None:
            self.stack[-1][2][1].__exit__(None, None, None); self.stack[-1][2] = None
        rf = record_function("A:" + label); rf.__enter__(); self.stack.append([label, rf, None])
        if label not in self.io:
            ins = [t for t in (a if isinstance(a, (tuple, list)) else (a,)) if torch.is_tensor(t)]
            self.io[label] = dict(inputs=[[list(t.shape), t.element_size()] for t in ins])

    def post(self, label, o):
        e = self.stack.pop(); assert e[0] == label, (e[0], label)
        if e[2] is not None: e[2][1].__exit__(None, None, None)
        e[1].__exit__(None, None, None)
        if "outputs" not in self.io[label]:
            os_ = o if isinstance(o, (tuple, list)) else (o,)
            self.io[label]["outputs"] = [[list(t.shape), t.element_size()] for t in os_ if torch.is_tensor(t)]
        if self.stack:
            g = f"{self.stack[-1][0]}/after:{label.split('.')[-1]}"
            rf = record_function("A:" + g); rf.__enter__(); self.stack[-1][2] = [g, rf]

    def remove(self):
        for h in self.handles: h.remove()


def op_group(label):
    """innermost annotation -> phase-7 target group"""
    m = re.match(r"interactions\.(\d+)(?:\.(\w+))?(?:/after:(\w+))?$", label)
    if m:
        i, sub, aft = m.groups()
        if sub == "conv_tp" or aft == "conv_tp_weights": return "conv(FlashTP)"   # after conv_tp_weights = conv glue
        if sub == "conv_tp_weights": return "embed.radial_MLP"
        if aft == "conv_tp": return "conv.glue_after(scatter/reshape)"
        if sub == "linear" or aft == "linear": return "node.interaction_linear(+avg,reshape)"
        if sub == "skip_tp" or aft == "skip_tp": return "node.skip_tp"
        if sub == "linear_up" or aft == "linear_up": return "node.linear_up"
        return "node.interaction_other"
    m = re.match(r"products\.(\d+)(?:\.(\w+))?(?:/after:(\w+))?$", label)
    if m:
        i, sub, aft = m.groups()
        if sub == "symmetric_contractions": return "SC(cueq/e3nn)"
        if sub == "linear" or aft in ("linear", "symmetric_contractions"): return "products.linear(+sc_add)"
        return "products.other"
    if label.startswith("readouts"): return "readouts"
    if label == "spherical_harmonics": return "embed.SH"
    if label == "radial_embedding": return "embed.radial_basis+cutoff"
    if label == "node_embedding": return "node_embedding"
    if label.startswith("model/after:"):
        a = label.split(":")[1]
        if a == "node_embedding": return "embed.edge_vectors/lengths(prepare_graph)"
        if a == "radial_embedding": return "embed.glue(after radial)"
        if a in ("spherical_harmonics",): return "embed.glue(after SH)"
        return "outputs/energy_sum/scale_shift"
    if label == "model": return "prepare_graph/e0"
    return label


def eager_sequence(trace_path):
    """ordered GPU activity list from the eager trace, each with its MACE-op label"""
    ev = json.load(open(trace_path))["traceEvents"]; ev = [e for e in ev if e.get("ph") == "X"]
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = {e["args"].get("correlation"): e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ua = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("A:")]
    cpuop = [e for e in ev if e.get("cat") == "cpu_op"]

    def by_tid(lst):
        d = collections.defaultdict(list)
        for e in lst: d[e["tid"]].append(e)
        for v in d.values(): v.sort(key=lambda e: e["ts"])
        return d
    ua_t, op_t = by_tid(ua), by_tid(cpuop)

    def innermost(d, tid, ts, pred=lambda e: True):
        best = None
        for e in d.get(tid, []):
            if e["ts"] > ts: break
            if e["ts"] <= ts <= e["ts"] + e["dur"] and pred(e):
                if best is None or e["dur"] <= best["dur"]: best = e
        return best
    seq2label = {}
    for e in cpuop:
        s = e.get("args", {}).get("Sequence number")
        if s is None or e["name"].startswith("autograd::engine"): continue
        a = innermost(ua_t, e["tid"], e["ts"])
        if a is not None and s not in seq2label: seq2label[s] = a["name"][2:]
    out = []
    for k in sorted(kern, key=lambda k: k["ts"]):
        r = rt.get(k["args"].get("correlation"))
        lab, bwd = "unknown", False
        if r is not None:
            evf = innermost(op_t, r["tid"], r["ts"], lambda e: e["name"].startswith("autograd::engine::evaluate_function"))
            if evf is not None:
                bwd = True; s = evf["args"].get("Sequence number")
                lab = seq2label.get(s, "bwd_unmapped:" + evf["name"].split(": ")[-1])
            else:
                a = innermost(ua_t, r["tid"], r["ts"]); lab = a["name"][2:] if a else "outside_model"
        out.append(dict(name=k["name"], cat=k["cat"], dur=k["dur"], label=lab, bwd=bwd))
    return out


def replay_sequence(trace_path):
    ev = json.load(open(trace_path))["traceEvents"]; ev = [e for e in ev if e.get("ph") == "X"]
    kern = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda k: k["ts"])
    return [dict(name=k["name"], cat=k["cat"], dur=k["dur"], ts=k["ts"]) for k in kern]


def norm(n):
    l = n.lower()
    if l.startswith("memcpy"): return "MEMCPY"
    if l.startswith("memset"): return "MEMSET"
    return n


def short(n, L=70):
    n = re.sub(r"<.*", "<...>", n) if len(n) > L else n
    return n[:L]


# ---------------------------------------------------------------- FLOP / byte estimates
def linear_flops_per_node(lin):
    """e3nn o3.Linear: 2 * sum_paths mul_in*mul_out*dim(ir)"""
    try:
        tot = 0
        for ins in lin.instructions:
            mi, mo = ins.path_shape
            tot += 2 * mi * mo * lin.irreps_out[ins.i_out].ir.dim
        return tot
    except Exception:
        return None


def mlp_flops_per_edge(mlp):
    tot = 0
    for p in mlp.parameters():
        if p.dim() == 2: tot += 2 * p.shape[0] * p.shape[1]
    return tot


def flop_table(model, N, E):
    f = {}
    for i, b in enumerate(model.interactions):
        for n in ("linear_up", "linear"):
            v = linear_flops_per_node(getattr(b, n))
            if v: f[f"L{i}.{n}"] = v * N
        st = b.skip_tp
        f[f"L{i}.skip_tp(approx, FullyConnectedTP w/ one-hot = per-species linear)"] = None
        f[f"L{i}.conv_tp_weights(radial MLP)"] = mlp_flops_per_edge(b.conv_tp_weights) * E
    for i, p in enumerate(model.products):
        v = linear_flops_per_node(p.linear)
        if v: f[f"P{i}.linear"] = v * N
    return f


def nbytes(lst):
    return sum(math.prod(s) * es for s, es in lst)


# ---------------------------------------------------------------- main
res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, torch=torch.__version__, systems={})
calcs, builders = {}, {}
for bk in BACK:
    conv, sc = dict(flash=("flashtp", "e3nn"), hybrid=("flashtp", "cueq"))[bk]
    calcs[bk], _ = HB.build_calc(conv=conv, sc=sc, check_indices=False)
    builders[bk] = ProtoBuilder(calcs[bk])
r_max = float(calcs[BACK[0]].r_max)


def save(): json.dump(res, open(out_path, "w"), indent=1)


for sname in SYS:
    atoms = load(sname)
    for bk in BACK:
        model = calcs[bk].models[0]
        try:
            bd = builders[bk].build(atoms)
            N, E = len(atoms), int(bd["edge_index"].shape[1])
            cap = round_cap(E); cap_tight = round_cap(E, 0.0)
            D = res["systems"].setdefault(sname, {})[bk] = dict(N=N, E=E, E_cap=cap, E_cap_tight=cap_tight)
            print(f"=== {sname} {bk} N={N} E={E} cap={cap}", flush=True)
            G = GraphedMACE(model, bd, cap, r_max)
            G.step(bd)
            D["t_replay"] = timeit(G.replay)
            D["t_copyin_replay"] = timeit(lambda: G.step(bd))
            # GPU profile of one replay
            for _ in range(3): G.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
                with record_function("REPLAY"):
                    t0 = time.perf_counter(); G.replay(); torch.cuda.synchronize(); twall = (time.perf_counter() - t0) * 1e3
            tr = f"{TMP}/_replay_{bk}_{sname}.json"; p.export_chrome_trace(tr)
            rep = replay_sequence(tr); os.remove(tr)
            D["replay_profiled_wall_ms"] = twall
            D["replay_n_gpu_ops"] = len(rep)
            D["replay_n_kernels"] = sum(1 for k in rep if k["cat"] == "kernel")
            D["replay_kernel_sum_us"] = sum(k["dur"] for k in rep)
            D["replay_busy_us"] = union_len([(k["ts"], k["ts"] + k["dur"]) for k in rep])
            D["replay_span_us"] = (rep[-1]["ts"] + rep[-1]["dur"] - rep[0]["ts"]) if rep else 0
            # eager annotated run on the SAME padded static inputs
            A = Annot(model)
            with torch_tools.default_dtype("float64"):
                for _ in range(2): model(G.static, compute_force=True, training=False)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
                    model(G.static, compute_force=True, training=False); torch.cuda.synchronize()
            A.remove()
            tr = f"{TMP}/_eager_{bk}_{sname}.json"; p.export_chrome_trace(tr)
            eag = eager_sequence(tr); os.remove(tr)
            # align
            ek = [e for e in eag]; ok = len(ek) == len(rep) and all(norm(a["name"]) == norm(b["name"]) for a, b in zip(ek, rep))
            D["align_ok"] = ok; D["eager_n_gpu_ops"] = len(ek)
            mm = [i for i, (a, b) in enumerate(zip(ek, rep)) if norm(a["name"]) != norm(b["name"])]
            D["align_first_mismatches"] = [(i, short(ek[i]["name"], 90), short(rep[i]["name"], 90)) for i in mm[:8]]
            D["n_name_mismatch"] = len(mm)
            D["multiset_equal"] = collections.Counter(norm(e["name"]) for e in ek) == collections.Counter(norm(e["name"]) for e in rep)
            if not ok:   # fall back: align by kernels only, greedy name match
                j = 0; lab = []
                for k in rep:
                    while j < len(ek) and norm(ek[j]["name"]) != norm(k["name"]): j += 1
                    if j < len(ek): lab.append((ek[j]["label"], ek[j]["bwd"])); j += 1
                    else: lab.append(("unaligned", False))
                D["align_unmatched"] = sum(1 for l in lab if l[0] == "unaligned")
            else:
                lab = [(e["label"], e["bwd"]) for e in ek]
            per = collections.defaultdict(lambda: dict(fwd_us=0.0, bwd_us=0.0, n_fwd=0, n_bwd=0, kernels=collections.Counter()))
            for k, (l, b) in zip(rep, lab):
                g = per[op_group(l)]
                if b: g["bwd_us"] += k["dur"]; g["n_bwd"] += 1
                else: g["fwd_us"] += k["dur"]; g["n_fwd"] += 1
                g["kernels"][short(k["name"]) + (" [bwd]" if b else "")] += k["dur"]
            D["per_op"] = {g: dict(fwd_us=v["fwd_us"], bwd_us=v["bwd_us"], n_fwd=v["n_fwd"], n_bwd=v["n_bwd"],
                                   top_kernels=[(n, round(t, 1)) for n, t in v["kernels"].most_common(5)]) for g, v in per.items()}
            tk = collections.Counter()
            for k in rep: tk[short(k["name"], 110)] += k["dur"]
            D["top_kernels"] = [(n, round(t, 1)) for n, t in tk.most_common(25)]
            D["io_bytes"] = {l: dict(in_MB=nbytes(v.get("inputs", [])) / 1e6, out_MB=nbytes(v.get("outputs", [])) / 1e6,
                                     inputs=v.get("inputs"), outputs=v.get("outputs")) for l, v in A.io.items()}
            D["flops"] = flop_table(model, N, cap)
            # padding cost: tight-cap graph
            del G; torch.cuda.empty_cache()
            G2 = GraphedMACE(model, bd, cap_tight, r_max); G2.step(bd)
            D["t_replay_tight_cap"] = timeit(G2.replay)
            del G2; torch.cuda.empty_cache()
            print(json.dumps(dict(replay=D["t_replay"]["median"], tight=D["t_replay_tight_cap"]["median"],
                                  busy_ms=D["replay_busy_us"] / 1e3, nk=D["replay_n_kernels"], align=ok)), flush=True)
        except Exception as ex:
            res["systems"].setdefault(sname, {})[bk] = dict(error=repr(ex), tb=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        save()
save(); print("DONE")
