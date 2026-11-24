"""md9 B3: sustained NVE MD throughput.  usage: b3bench.py <providers,comma> <systems,comma> [--warm 10] [--steps 200]
providers: stock (vv.py + StockEager), ase (ase VelocityVerlet + stock MACECalculator, default ctor => stress computed
for periodic cells), graphed (vv.py + GraphedProto node-hybrid), gpustep (vv.py + int9 GPUStep), gpustep_cg (same, integration
before/after C's step graph captured as two small CUDA graphs: vv.capture_split).
Boundary: one MD step = integration + force evaluation + whatever host interaction the provider requires; Epot/Ekin
recorded on GPU every step (ASE: no observer). Timing: warmup steps, then `steps` steps split in 4 blocks of steps/4 with
torch.cuda.synchronize() only at block boundaries; ms/step per block -> median [min-max] over blocks, and overall mean.
Afterwards (untimed) a sync audit: 10 steps under torch.cuda.set_sync_debug_mode('warn') (count of synchronizing CUDA
calls per step as reported by torch) + the provider's own explicit sync counters.
Must run under scripts/md9/timing_job.sh (TIMING_LOCK)."""
import os, sys, json, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vv as V  # first
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("providers"); ap.add_argument("systems")
ap.add_argument("--warm", type=int, default=10); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--out", default=None)
args = ap.parse_args()
OUT = args.out or f"{V.R}/results/md9/b3_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
res = dict(V.env_info(), warm=args.warm, steps=args.steps, dt_fs=args.dt, rows=[])
_stock = {}


def stock_calc():
    if "c" not in _stock:
        _stock["c"] = V.stock_calc()
    return _stock["c"]


def timed(step, nblk=4):
    per = args.steps // nblk; blk = []
    torch.cuda.synchronize(); t00 = time.perf_counter()
    for _ in range(nblk):
        t0 = time.perf_counter()
        for _ in range(per):
            step()
        torch.cuda.synchronize(); blk.append((time.perf_counter() - t0) / per * 1e3)
    tot = (time.perf_counter() - t00) / (per * nblk) * 1e3
    return dict(ms_step_mean=tot, ms_step_blocks=blk, ms_step_median=float(np.median(blk)),
                ms_step_min=float(min(blk)), ms_step_max=float(max(blk)), timed_steps=per * nblk)


def one(pk, name):
    atoms = V.load_start(name); N = len(atoms)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    row = dict(provider=pk, system=name, N=N, periodic=bool(any(atoms.pbc)))
    t0 = time.perf_counter()
    if pk == "ase":
        from ase.md.verlet import VelocityVerlet
        a = atoms.copy(); a.calc = stock_calc()
        dyn = VelocityVerlet(a, timestep=args.dt * V.units.fs)
        a.get_forces(); torch.cuda.synchronize(); row["setup_first_force_s"] = time.perf_counter() - t0
        step = lambda: dyn.run(1)
        prov_syncs = None; md = None
        row["note"] = "ASE VelocityVerlet + MACECalculator(default_dtype=float64, compute_stress default)"
    else:
        kind = {"stock": "stock", "graphed": "graphed", "gpustep": "gpustep", "gpustep_cg": "gpustep"}[pk]
        kw = dict(calc=stock_calc()) if kind == "stock" else {}
        prov = V.make_provider(kind, atoms, stress=False, **kw)
        n_max = args.warm + args.steps + 40
        md = V.VelocityVerletGPU(prov, atoms, args.dt, n_max, check_every=10 if kind == "gpustep" else 0)
        torch.cuda.synchronize(); row["setup_first_force_s"] = time.perf_counter() - t0
        if pk == "gpustep_cg":
            t1 = time.perf_counter(); md.capture_split(); torch.cuda.synchronize(); row["vv_capture_s"] = time.perf_counter() - t1
        step = md.step; prov_syncs = prov.syncs
        row["provider_name"] = prov.name
    t1 = time.perf_counter()
    for _ in range(args.warm):
        step()
    torch.cuda.synchronize(); row["warmup_s"] = time.perf_counter() - t1
    if prov_syncs is not None:
        before = {k: v for k, v in prov_syncs.items()}
    row.update(timed(step))
    if prov_syncs is not None:
        row["explicit_syncs_per_step"] = {k: (prov_syncs[k] - before.get(k, 0)) / row["timed_steps"] for k in prov_syncs}
    row["atom_steps_per_s"] = N / (row["ms_step_median"] * 1e-3)
    row["ns_per_day_at_dt"] = args.dt * 1e-6 * 86400 / (row["ms_step_median"] * 1e-3)
    row["peak_mem_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    if md is not None and hasattr(md.prov, "runner"):
        row["runner_stats"] = md.prov.runner.stats()
    if md is not None and hasattr(md.prov, "sim") and hasattr(md.prov.sim, "stats"):
        try:
            row["gpustep_stats"] = md.prov.sim.stats()
        except Exception as e:
            row["gpustep_stats"] = repr(e)
    try:
        row["sync_audit"] = V.audit_syncs(step, 10)
    except Exception as e:
        row["sync_audit"] = repr(e)
    if md is not None:
        ep, ek = md.energies(); et = (ep + ek) / N
        row["Etot_per_atom_range_during_bench"] = float(et.max() - et.min()); row["finite"] = bool(np.isfinite(et).all())
        row["check_failures"] = md.check_fail; row["n_rollbacks"] = md.n_rollbacks
        row["n_post_captures"] = md.n_post_captures
    print(f"{pk:11s} {name:10s} N={N:5d} {row['ms_step_median']:9.3f} ms/step [{row['ms_step_min']:.3f}-"
          f"{row['ms_step_max']:.3f}] {row['atom_steps_per_s']:.3e} atom-steps/s syncs/step(audit)="
          f"{row['sync_audit']['per_step'] if isinstance(row['sync_audit'], dict) else row['sync_audit']} "
          f"peak={row['peak_mem_MiB']:.0f}MiB", flush=True)
    return row


for name in args.systems.split(","):
    for pk in args.providers.split(","):
        try:
            res["rows"].append(one(pk, name))
        except Exception as e:
            traceback.print_exc(); res["rows"].append(dict(provider=pk, system=name, error=repr(e)[:500]))
            print(f"ERROR {pk} {name} {e!r}"[:300], flush=True)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        torch.cuda.empty_cache()
res["gpu_end"] = V.gpu_info()
json.dump(res, open(OUT, "w"), indent=1, default=float)
