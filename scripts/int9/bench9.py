"""int9 B1/B2 timing.  usage: python -I bench9.py OUT.json configs(comma) systems(comma)
configs: stock | hybrid+embed+node | nodehybrid+embed+node | gpustep[:<nl>] (Phase-9 GPUStep, node conv with external
  CSR; nl = auto|cell|brute) | gpustep_hyb[:<nl>] (same whole-step graph with the FlashTP-hybrid model; in-model conv
  needs no CSR)
Boundaries (all synced with torch.cuda.synchronize after each rep; median [p10-p90]):
  stock  B1 = eager model(batch, compute_force) on a prebuilt stock batch (no graph exists for stock);
         B2 = calc.calculate(atoms) = host matscipy NL + batch + H2D + eager fwd/bwd + D2H (stock has no GPU-resident path)
  graphed (GraphedRunner, Phase 7/8):
         B1 = graph.replay() with prebuilt neighbour graph already loaded
         B2 = B2gpu: ProtoGPU.build(pos_gpu) (O(N^2) brute force on GPU, nonzero() -> 1+ host syncs, wraps on GPU,
              no H2D of positions) + _Graph.load (copy edges/shifts into static buffers, pad) + replay
         B2host (reference, = Phase 8 'full step'): runner.step_tensors(atoms) = host proto (pinned H2D) + load + replay
  gpustep (Phase 9): B1 = graph replay of the whole captured step (NL + model) with positions already in the static buffer
         = also the complete step; B2 = sim.forces(pos_gpu) (D2D copy into static positions + replay + async flag copy);
         B2chk = forces + check() every step (one host sync per step, worst case)
Stock reps adaptive: if one call > 0.2 s -> 2 warmups, 10 reps (recorded)."""
import os, sys, json, time, traceback
R = os.environ["R"]
sys.path.insert(0, f"{R}/scripts/int9")
import common9 as C
import numpy as np
import torch

out_path = sys.argv[1]; CFG = sys.argv[2].split(","); SYS = sys.argv[3].split(",")
res = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, gpu_start=C.gpuinfo(), warm=C.WARM, reps=C.REPS,
           torch=torch.__version__, configs={}, systems={})
built = {}
for c in CFG:
    base = ("hybrid+embed+node" if c.startswith("gpustep_hyb") else "nodeext+embed+node") if c.startswith("gpustep") else c
    if base in built:
        built[c] = built[base]; continue
    t0 = time.perf_counter(); built[c] = C.build(base); torch.cuda.synchronize()
    built[base] = built[c]
    res["configs"][c] = dict(build_install_s=time.perf_counter() - t0)
    print("built", c, res["configs"][c], flush=True)


def adaptive(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); dt = time.perf_counter() - t0
    if dt > 0.2:
        return C.timeit(fn, warm=2, reps=10)
    return C.timeit(fn)


def run_one(c, atoms):
    calc, model = built[c]
    D = {}
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    if c == "stock":
        from mace.tools import torch_tools
        bd = C.stock_batch(calc, atoms)
        bd["positions"].requires_grad_(False)
        D["E"] = int(bd["edge_index"].shape[1])

        def b1():
            with torch_tools.default_dtype("float64"):
                model(dict(bd), compute_force=True, compute_stress=False, training=False)
        D["B1"] = adaptive(b1)
        a2 = atoms.copy()
        D["B2"] = adaptive(lambda: calc.calculate(a2, ["energy", "forces"]))
    elif c.startswith("gpustep"):
        import gpu_step as GS
        nl = c.split(":")[1] if ":" in c else "auto"
        t0 = time.perf_counter()
        sim = GS.GPUStep(calc, model, atoms, nl=nl)
        pos = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
        sim.forces(pos); ok = sim.check(); torch.cuda.synchronize()
        D["first_step_s"] = time.perf_counter() - t0
        D["check"] = ok
        st = sim.stats(); D.update(E=st["n_edges_last"], E_cap=st["E_cap"], capture_s=st["last_capture_s"], nl=st["nl"],
                                  path=st["path"])
        D["B1"] = C.timeit(sim.replay)
        D["B2"] = C.timeit(lambda: sim.forces(pos))
        D["B2chk"] = C.timeit(lambda: (sim.forces(pos), sim.check()))
        D["stats"] = sim.stats()
        del sim
    else:
        rn = C.GR.GraphedRunner(calc, model)
        t0 = time.perf_counter(); o = rn.step_tensors(atoms); torch.cuda.synchronize()
        D["first_step_s"] = time.perf_counter() - t0
        key = rn.key_of(atoms); g = rn.graphs[key]
        D.update(E=o["E"], E_cap=o["E_cap"], capture_s=rn.stats()["last_capture_s"])
        D["B1"] = C.timeit(g.replay)
        pg = C.ProtoGPU(calc, atoms)
        pos = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
        bd = pg.build(pos)
        assert int(bd["edge_index"].shape[1]) == o["E"], (bd["edge_index"].shape, o["E"])

        def b2():
            b = pg.build(pos)
            return rn._step_bd(key, atoms, b)
        D["B2"] = C.timeit(b2)
        D["B2_nl_only"] = C.timeit(lambda: pg.build(pos))
        D["B2host"] = C.timeit(lambda: rn.step_tensors(atoms))
        del rn, g
    torch.cuda.synchronize()
    D["peak_MiB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
    return D


for si, s in enumerate(SYS):
    atoms = C.load(s)
    order = CFG[si % len(CFG):] + CFG[:si % len(CFG)]
    S = res["systems"][s] = dict(N=len(atoms), pbc=bool(atoms.pbc.any()), order=order)
    for c in order:
        try:
            D = run_one(c, atoms)
        except torch.cuda.OutOfMemoryError as ex:
            D = dict(error=f"OOM: {str(ex)[:300]}")
        except Exception as ex:  # noqa: BLE001
            D = dict(error=f"{type(ex).__name__}: {ex}"[:500], tb=traceback.format_exc()[-3000:])
            print(D["tb"], flush=True)
        S[c] = D
        torch.cuda.empty_cache()
        brief = {k: (round(v["median"], 4) if isinstance(v, dict) and "median" in v else v) for k, v in D.items()
                 if k in ("B1", "B2", "B2chk", "B2host", "B2_nl_only", "E", "E_cap", "peak_MiB", "first_step_s", "error")}
        print(s, c, json.dumps(brief), flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
res["gpu_end"] = C.gpuinfo()
json.dump(res, open(out_path, "w"), indent=1)
print("DONE", flush=True)
