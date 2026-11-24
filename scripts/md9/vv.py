"""md9: GPU-resident velocity-Verlet (NVE) with pluggable force providers.

Units = ASE (eV, A, amu, ASE time unit; dt = dt_fs * ase.units.fs). State on GPU in fp64: positions x [N,3],
momenta p [N,3] (= ASE momenta, amu*A/ase_t), masses m [N,1] (ASE atomic masses). Algorithm = ase.md.verlet.VelocityVerlet
(momentum form, no constraints):
    p += 0.5 dt F ;  x += dt p/m ;  (E, F) = force_fn(x) ;  p += 0.5 dt F
Epot (provider energy) and Ekin = sum p^2/(2m) are written into preallocated GPU buffers at a device-side step index
(no host sync). Optional position snapshots every `save_every` steps are kept on the GPU.

Force provider interface:  provider(x) -> (E 0-d tensor, F [N,3]) on GPU, fp64; F may alias a provider buffer that is
valid until the next provider call (VV consumes it before that). Attributes:
    provider.name, provider.syncs (dict: explicit host syncs per category, counted by the provider itself),
    provider.capturable (bool: whole VV step may be captured in a CUDA graph), provider.last_stress (Voigt or None),
    provider.check() (optional; called by VV every `check_every` steps; counted as a sync),
    provider.edges() (optional; last neighbour list, host numpy: (edge_index [2,E], unit_shifts [E,3])) -- harness only.

Providers
 (a) StockEager  : stock fp64 MACE model on the stock batch (calc._atoms_to_batch, matscipy NL on CPU) -- ASE-free
                   reference; positions D2H each step (1 explicit sync) + the stock path's own H2D/implicit syncs.
 (b) GraphedProto: graphed7 GraphedRunner + cfg8 node-hybrid (+embed+node) + ProtoBuilder; positions D2H each step
                   into a host Atoms (1 explicit sync), proto builder then does its own H2D + nonzero() syncs.
 (c) GPUStepProv : agent C's scripts/int9/gpu_step.GPUStep; forces(pos_gpu) GPU-resident; check() every k steps.
Import this module BEFORE torch/mace elsewhere (graphed_runner pins flashTP).
"""
import os
import sys
import time
import warnings
import importlib

R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
for _p in (f"{R}/scripts/graphed7", f"{R}/scripts/prof8"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import graphed_runner as GR  # noqa: E402  (first)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms, units  # noqa: E402
from mace.tools import torch_tools  # noqa: E402

MODEL = f"{R}/models/MACE-OFF23_medium.model"


def voigt(s):
    s = s.detach().reshape(3, 3).cpu().numpy()
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def load_start(name):
    st = np.load(f"{R}/results/md9/systems/start_{name}.npz")
    a = Atoms(numbers=st["numbers"], positions=st["positions"], cell=st["cell"], pbc=st["pbc"])
    a.set_masses(st["masses"]); a.set_velocities(st["velocities"])
    return a


def stock_calc():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64")


# --------------------------------------------------------------------------------------------- providers
class StockEager:
    """(a) reference: stock fp64 model, stock batch (CPU matscipy NL) per step."""
    capturable = False

    def __init__(self, atoms, stress=False, calc=None, keep_edges=False):
        self.name = "stock_eager"
        self.calc = calc if calc is not None else stock_calc()
        self.model = self.calc.models[0]
        self.atoms = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        self.stress = bool(stress) and any(atoms.pbc)
        self.keep_edges = keep_edges
        self.syncs = dict(d2h_positions=0)
        self.last_stress = None
        self._bd = None

    def eval_np(self, pos_np, stress=None):
        self.atoms.arrays["positions"][:] = pos_np
        bd = GR._stock_build(self.calc, self.atoms)
        st = self.stress if stress is None else stress
        with torch_tools.default_dtype("float64"):
            o = self.model(bd, compute_force=True, compute_stress=st, training=False)
        if self.keep_edges:
            self._bd = bd
        self.last_stress = o["stress"] if st else None
        return o["energy"].detach()[0], o["forces"].detach()

    def __call__(self, x):
        self.syncs["d2h_positions"] += 1
        return self.eval_np(x.detach().cpu().numpy())

    def edges(self):
        bd = self._bd
        return bd["edge_index"].cpu().numpy(), bd["unit_shifts"].cpu().numpy()


class GraphedProto:
    """(b) graphed node-hybrid (Phase 8) with the proto GPU builder; needs atoms on the host."""
    capturable = False

    def __init__(self, atoms, cfg="nodehybrid+embed+node", stress=False, keep_edges=False):
        import cfg8
        self.name = f"graphed_{cfg}_proto"
        self.calc, self.model = cfg8.build(cfg)
        self.stress = bool(stress) and any(atoms.pbc)
        self.runner = GR.GraphedRunner(self.calc, self.model, stress=self.stress, builder="proto")
        self.atoms = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        self.syncs = dict(d2h_positions=0)
        self.keep_edges = keep_edges
        self.last_stress = None
        self._bd = None
        if keep_edges:
            b0 = self.runner._build

            def b(a):
                bd = b0(a); self._bd = bd; return bd
            self.runner._build = b

    def __call__(self, x):
        self.syncs["d2h_positions"] += 1
        self.atoms.arrays["positions"][:] = x.detach().cpu().numpy()
        o = self.runner.step_tensors(self.atoms)
        self.last_stress = o["stress"]
        return o["energy"][0], o["forces"]

    def edges(self):
        bd = self._bd
        return bd["edge_index"].cpu().numpy(), bd["unit_shifts"].cpu().numpy()


class GPUStepProv:
    """(c) agent C's GPUStep (scripts/int9/gpu_step.py): whole step (NL + model) = one CUDA graph, positions D2D-copied
    into the graph's static buffer, zero host syncs per step; check() (1 event sync) every `check_every` VV steps."""
    capturable = False   # forces() launches C's graph; nesting a graph launch inside a capture is not attempted

    def __init__(self, atoms, stress=False, cfg="nodeext+embed+node", nl=None, **kw):
        nl = nl or os.environ.get("MD9_NL", "auto")
        if f"{R}/scripts/int9" not in sys.path:
            sys.path.insert(0, f"{R}/scripts/int9")
        import common9 as C9
        import gpu_step as GS
        self.calc, self.model = C9.build(cfg)
        self.stress = bool(stress) and any(atoms.pbc)
        self.sim = GS.GPUStep(self.calc, self.model, atoms, stress=self.stress, nl=nl, **kw)
        self.name = f"gpustep_{cfg}_{self.sim.nl_name}_{self.sim.path}"
        self.syncs = dict(check=0)
        self.last_stress = None

    def __call__(self, x):
        o = self.sim.forces(x)
        if self.stress:
            self.last_stress = o[2]
        return o[0].reshape(()), o[1]

    def check(self):
        self.syncs["check"] += 1
        return self.sim.check()

    def edges(self):
        if self.sim.path != "graph":      # int9 proto fallback (unsupported NL geometry): edges not exposed
            return None
        ei, us = self.sim.edges()
        return ei.detach().cpu().numpy(), us.detach().cpu().numpy()


# --------------------------------------------------------------------------------------------- integrator
class VelocityVerletGPU:
    def __init__(self, provider, atoms, dt_fs, n_max, save_every=0, check_every=0):
        dev = torch.device("cuda")
        self.prov = provider
        self.N = len(atoms)
        self.dt = float(dt_fs) * units.fs
        self.dt_fs = float(dt_fs)
        self.x = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=dev).contiguous()
        self.p = torch.tensor(atoms.get_momenta(), dtype=torch.float64, device=dev).contiguous()
        self.m = torch.tensor(atoms.get_masses(), dtype=torch.float64, device=dev).reshape(-1, 1).contiguous()
        self.epot = torch.zeros(n_max + 1, dtype=torch.float64, device=dev)
        self.ekin = torch.zeros(n_max + 1, dtype=torch.float64, device=dev)
        self.idx = torch.zeros((), dtype=torch.int64, device=dev)    # device-side step counter
        self.one = torch.ones((), dtype=torch.int64, device=dev)
        self.n_max = n_max
        self.save_every = int(save_every)
        self.check_every = int(check_every)
        self.snaps, self.snap_steps = [], []
        self.nstep = 0                                                  # host-side counter (no sync)
        self.check_fail = []
        self.graph = None
        self.g_pre = self.g_post = None
        self._post_key = None
        self.n_post_captures = 0
        E, F = provider(self.x)
        self.F = F.detach().clone()                                     # static VV force buffer
        self._record(E.detach())
        if self.save_every:
            self._snap()
        self.n_rollbacks = 0
        if self.check_every and hasattr(provider, "check"):
            rep = provider.check()
            if not self._ok(rep):
                raise RuntimeError(f"initial check failed: {rep}")
            self._save_ckpt()

    def _record(self, E):
        self.epot.index_copy_(0, self.idx.view(1), E.reshape(1).to(torch.float64))
        ek = (0.5 * (self.p * self.p) / self.m).sum().reshape(1)
        self.ekin.index_copy_(0, self.idx.view(1), ek)

    def _snap(self):
        self.snaps.append(torch.stack((self.x.clone(), self.p / self.m)))
        self.snap_steps.append(self.nstep)

    def _pre(self):
        self.p.add_(self.F, alpha=0.5 * self.dt)
        self.x.add_(self.p / self.m, alpha=self.dt)

    def _post(self, E, F):
        self.F.copy_(F)
        self.p.add_(self.F, alpha=0.5 * self.dt)
        self.idx.add_(self.one)
        self._record(E)

    def _body(self):
        self._pre()
        E, F = self.prov(self.x)
        self._post(E.detach(), F.detach())

    def capture(self, n_warm=3):
        """Capture one whole VV step (integration + forces) in one CUDA graph (provider.capturable required)."""
        if not getattr(self.prov, "capturable", False):
            raise RuntimeError(f"provider {self.prov.name} not capturable")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                self._body(); self.nstep += 1
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._body()
        self.graph = g          # capture does not execute; the captured step runs at the next step()

    def capture_split(self):
        """Provider that is itself one CUDA graph with static outputs (int9 GPUStep): capture the integration
        before (half kick + drift) and after (half kick + record) the provider call as two small graphs ->
        per step: replay(pre) + provider (D2D pos copy + its graph) + replay(post). The post graph is keyed on the
        provider's output pointers and recaptured if they change (e.g. after an E_cap recapture)."""
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._pre()
        self.g_pre = g

    def _split_step(self):
        self.g_pre.replay()
        E, F = self.prov(self.x)
        E, F = E.detach(), F.detach()
        key = (E.data_ptr(), F.data_ptr())
        if key != self._post_key:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                self._post(E, F)
            self.g_post, self._post_key = g, key
            self.n_post_captures += 1
        self.g_post.replay()

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        elif self.g_pre is not None:
            self._split_step()
        else:
            self._body()
        self.nstep += 1
        if self.save_every and self.nstep % self.save_every == 0:
            self._snap()
        if self.check_every and self.nstep % self.check_every == 0 and hasattr(self.prov, "check"):
            self._check()

    @staticmethod
    def _ok(rep):
        return rep.get("ok", True) if isinstance(rep, dict) else (rep is None or bool(rep))

    def _save_ckpt(self):
        self.ckpt = (self.x.clone(), self.p.clone(), self.F.clone(), self.idx.clone(), self.nstep, len(self.snaps))

    def _check(self, depth=0):
        rep = self.prov.check()
        if isinstance(rep, dict) and (rep.get("recaptured") or not rep.get("ok", True)):
            self.check_fail.append(dict(step=self.nstep, rep=dict(rep)))
        if self._ok(rep):
            self._save_ckpt(); return
        # overflow: results since the last ok check are invalid (NaN) -> roll back to the checkpoint and redo
        if depth >= 4:
            raise RuntimeError(f"overflow persists after {depth} rollbacks: {rep}")
        self.n_rollbacks += 1
        target = self.nstep
        x, p, F, idx, n, ns = self.ckpt
        self.x.copy_(x); self.p.copy_(p); self.F.copy_(F); self.idx.copy_(idx); self.nstep = n
        del self.snaps[ns:]; del self.snap_steps[ns:]
        while self.nstep < target:
            self._split_step() if self.g_pre is not None else self._body()
            self.nstep += 1
            if self.save_every and self.nstep % self.save_every == 0:
                self._snap()
        self._check(depth + 1)

    def run(self, n):
        for _ in range(n):
            self.step()

    def energies(self):
        n = self.nstep + 1
        return self.epot[:n].cpu().numpy(), self.ekin[:n].cpu().numpy()

    def snapshots(self):
        if not self.snaps:
            return None, None
        S = torch.stack(self.snaps).cpu().numpy()
        return S, np.array(self.snap_steps)


def make_provider(kind, atoms, stress=False, keep_edges=False, **kw):
    if kind == "stock":
        return StockEager(atoms, stress=stress, keep_edges=keep_edges, **kw)
    if kind == "graphed":
        return GraphedProto(atoms, stress=stress, keep_edges=keep_edges, **kw)
    if kind == "gpustep":
        return GPUStepProv(atoms, stress=stress, **kw)
    raise ValueError(kind)


def audit_syncs(fn, n):
    """Run fn() n times under torch.cuda.set_sync_debug_mode('warn'); return #synchronizing-op warnings per call
    and the distinct messages (first 120 chars). Not used inside timed loops."""
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for _ in range(n):
                fn()
            torch.cuda.set_sync_debug_mode("default")
        finally:
            torch.cuda.set_sync_debug_mode("default")
    msgs = [str(x.message) for x in w if "synchroniz" in str(x.message).lower()]
    kinds = {}
    for m in msgs:
        k = m.split("\n")[0][:120]; kinds[k] = kinds.get(k, 0) + 1
    return dict(per_step=len(msgs) / n, n_steps=n, kinds={k: v / n for k, v in kinds.items()})


def gpu_info():
    import subprocess
    q = "index,name,uuid,clocks.sm,clocks.max.sm"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()


def env_info():
    return dict(slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, torch=torch.__version__,
                gpu=gpu_info(), date=time.strftime("%Y-%m-%dT%H:%M:%S"))
