"""int9 shared helpers: backends, systems, timing, GPU-resident proto builder (ProtoGPU).
Import order: graphed_runner first (pins flashTP_e3nn), then torch/mace."""
import os, sys, time, json, subprocess
R = os.environ["R"]
for p in (f"{R}/scripts/graphed7", f"{R}/scripts/prof8", f"{R}/scripts/graph", f"{R}/scripts/int9"):
    if p not in sys.path:
        sys.path.insert(0, p)
import graphed_runner as GR  # noqa: E402  (first)
import cfg8  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ase.io  # noqa: E402
from proto_graph import ProtoBuilder, mace_cells, DYN  # noqa: E402

SYSDIR = f"{R}/results/int9/systems"
WARM, REPS = int(os.environ.get("WARM", 10)), int(os.environ.get("REPS", 60))


def load(name):
    return ase.io.read(f"{SYSDIR}/{name}.extxyz")


def stats(ts):
    a = np.asarray(ts)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), n=int(a.size))


def timeit(fn, warm=None, reps=None):
    warm = WARM if warm is None else warm; reps = REPS if reps is None else reps
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return stats(ts)


def gpuinfo():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.max.sm,temperature.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def build(cfg):
    """cfg: 'stock' | 'hybrid+embed+node' | 'nodehybrid+embed+node' | 'nodeext+embed+node' (node conv with external CSR)."""
    if cfg == "stock":
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64")
        return calc, calc.models[0]
    if cfg.startswith("nodeext"):
        import nodeconv_ext as NX
        calc, info = GR.HB.build_calc(conv="e3nn", sc="cueq", check_indices=False)
        model = calc.models[0]
        handles = [NX.install(model)]
        for t in cfg8._tr(cfg.split("+")):
            handles.append(t(model))
        calc.graphed7_kind = "nodeext"; calc.graphed7_info = info; calc.graphed7_transform_handles = handles
        return calc, model
    return cfg8.build(cfg)


def stock_batch(calc, atoms):
    return GR._stock_build(calc, atoms)


class ProtoGPU:
    """GPU-resident variant of proto_graph.ProtoBuilder for a fixed template (numbers, pbc, cell):
    build(pos_gpu) -> MACE batch dict.  Same neighbour_list() code (O(N^2) brute force, nonzero() -> host sync) but
    positions are NOT copied from the host and the periodic integer wraps are computed on the GPU
    (floor(pos @ inv(ext_cell)) along periodic axes) instead of numpy. Supports pbc all-False or all-True only.
    cell/rcell/volume: constant per template (periodic: = stock values; non-periodic: computed once from the template
    positions -- the model never reads them when stress is off)."""

    def __init__(self, calc, atoms):
        self.pb = ProtoBuilder(calc)
        self.pb._check_supported(atoms)
        self.t = self.pb._template(atoms)
        self.pbc = self.t["pbc"]
        if any(self.pbc) and not all(self.pbc):
            raise NotImplementedError("mixed pbc")
        dev = self.pb.device
        ret, ext = mace_cells(atoms.positions, self.pbc, np.array(atoms.get_cell()), self.pb.cutoff)
        self.ext_np = ext
        cell_t = torch.tensor(ret, dtype=torch.float64)
        vol = torch.linalg.det(cell_t)
        rcell = 2 * torch.pi * torch.linalg.inv(cell_t.mT) if torch.abs(vol) > 0 else torch.zeros(3, 3, dtype=torch.float64)
        self.cell = cell_t.to(dev); self.rcell = rcell.to(dev); self.vol = vol.reshape(1).to(dev)
        self.ext = torch.tensor(ext, dtype=torch.float64, device=dev)
        self.inv_ext = torch.tensor(np.linalg.inv(ext), dtype=torch.float64, device=dev)

    def build(self, pos):
        w = torch.floor(pos @ self.inv_ext) if any(self.pbc) else None
        ei, us, sh = self.pb.neighbour_list(pos, self.pbc, self.ext_np, self.ext, self.cell, w)
        dyn = dict(positions=pos, edge_index=ei, shifts=sh, unit_shifts=us, cell=self.cell, rcell=self.rcell, volume=self.vol)
        st = self.t["static"]
        return {k: (dyn[k] if k in dyn else st[k]) for k in self.t["keys"]}


def edge_set(ei, us):
    """sorted host array of (src, dst, us0, us1, us2) rows for bitwise neighbour-set comparison."""
    a = np.concatenate([ei.detach().t().cpu().numpy().astype(np.int64), np.rint(us.detach().cpu().numpy()).astype(np.int64)], 1)
    if a.size:
        a = a[np.lexsort(a.T[::-1])]
    return a
