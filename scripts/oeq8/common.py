"""oeq8 shared helpers: model load, captured-input IO, e3nn reference conv, error metrics, timers."""
import os, sys, time, json, contextlib
import numpy as np
import torch

R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
MODEL = f"{R}/models/MACE-OFF23_medium.model"
SYSDIR = f"{R}/results/workloads/systems"
CAPDIR = os.environ.get("OEQ8_CAPDIR", f"{R}/tmp/oeq8cap")
SYSTEMS = ["aspirin", "water192", "org192", "water648", "org648", "wbox648"]
sync = torch.cuda.synchronize


def load_calc(**kw):
    from mace.calculators import MACECalculator
    before = torch.get_default_dtype()
    calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False, **kw)
    torch.set_default_dtype(before)
    return calc


def load_atoms(name):
    from ase.io import read
    return read(f"{SYSDIR}/{name}.extxyz")


def gen_grad(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64).cuda()


def load_cap(name, layer):
    d = torch.load(f"{CAPDIR}/{name}_L{layer}.pt", map_location="cuda")
    return d


def ref_conv(conv_tp, x, y, w, ei):
    """MACE stock: conv_tp(gather) + scatter_sum (blocks.py semantics)."""
    from mace.tools.scatter import scatter_sum
    mji = conv_tp(x[ei[0]], y, w)
    return scatter_sum(src=mji, index=ei[1], dim=0, dim_size=x.shape[0])


def fwd_bwd(fn, x, y, w, g):
    xs, ys, ws = (t.detach().clone().requires_grad_(True) for t in (x, y, w))
    out = fn(xs, ys, ws)
    gx, gy, gw = torch.autograd.grad(out, (xs, ys, ws), g)
    return out.detach(), gx, gy, gw


def err(a, ref):
    a = a.double(); ref = ref.double()
    d = (a - ref).abs()
    mref = ref.abs().max().item() if ref.numel() else 0.0
    mab = d.max().item() if d.numel() else 0.0
    nz = ref.abs() < 1e-8
    return dict(max_abs=mab, scaled=(mab / mref if mref > 0 else mab), max_ref=mref,
                n_nearzero=int(nz.sum().item()),
                max_abs_nearzero=(d[nz].max().item() if nz.any() else 0.0),
                bitwise_equal=bool(torch.equal(a, ref)))


def stats(ts):
    a = np.asarray(ts)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), n=len(a))


def time_eager(fn, warm=10, reps=50):
    for _ in range(warm): fn()
    sync(); ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return stats(ts)


def time_graph(fn, warm=10, reps=50, pre=None):
    """Capture fn into a CUDA graph and time replays with events. pre() runs outside capture before capturing."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(warm): g.replay()
    sync()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record(); g.replay(); e1.record(); sync(); ts.append(e0.elapsed_time(e1))
    del g
    return stats(ts)


def bench_conv(fn, x, y, w, g, graph=True):
    """fn(x,y,w)->out. Returns dict of eager/graph timings for fwd, bwd, fwd+bwd."""
    res = {}
    xs, ys, ws = (t.detach().clone().requires_grad_(True) for t in (x, y, w))

    def f_fwd():
        with torch.no_grad():
            return fn(x, y, w)

    def f_fb():
        o = fn(xs, ys, ws)
        return torch.autograd.grad(o, (xs, ys, ws), g)

    out_keep = fn(xs, ys, ws)

    def f_bwd():
        return torch.autograd.grad(out_keep, (xs, ys, ws), g, retain_graph=True)

    for k, f in (("fwd", f_fwd), ("bwd", f_bwd), ("fwd_bwd", f_fb)):
        try:
            res[f"eager_{k}"] = time_eager(f)
        except Exception as e:  # noqa
            res[f"eager_{k}"] = dict(error=repr(e)[:300])
        if graph:
            try:
                res[f"graph_{k}"] = time_graph(f)
            except Exception as e:  # noqa
                res[f"graph_{k}"] = dict(error=repr(e)[:300])
                torch.cuda.synchronize()
    return res


@contextlib.contextmanager
def quiet():
    sys.stdout.flush(); saved = os.dup(1); nul = os.open(os.devnull, os.O_WRONLY); os.dup2(nul, 1)
    try: yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(nul); os.close(saved)


def env_info():
    import subprocess
    try:
        uuid = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        uuid = "?"
    return dict(job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, gpu=uuid, torch=torch.__version__,
                python=sys.executable)
