"""graphed7: corrected CUDA-graphed MACE runner (whole energy + force [+ stress] call captured as one CUDA graph).

Derived from scripts/cudagraph6/graphed_model.py (copied, not imported; the prototype is left untouched).

Fixes vs the cudagraph6 prototype (validate6 findings 3e, 3f)
  * Every graph is keyed by (n_atoms, atomic numbers in order, pbc, cell-is-zero, head, dtype, stress flag).
    A key mismatch NEVER replays an existing graph: the runner transparently builds a separate graph for the new
    key and keeps up to `max_graphs` (default 4) graphs in an LRU dict (least recently used one is freed).
  * Per call the key is recomputed from atoms.numbers.tobytes() etc. (host only, no GPU sync) and the stored
    numbers of the selected graph are compared again (np.array_equal) before replay.
  * Per-node static tensors (node_attrs, atomic_numbers, batch, ptr, head, pbc, ...) are owned per graph; if the
    batch dict passes different tensor objects than those captured from, their values are copied in (shape-checked),
    so even a builder that hands out new statics cannot leave stale species in the graph.
  * Recapture (E > E_cap) builds a brand-new _Graph object (all bookkeeping: N, E_cap, padding indices, static
    buffers, node_attrs, ptr/batch, outputs) and frees the old one. New cap = round_cap(E, 0.25).

Padding (unchanged design): real edges in [0,E), padded edge k is a self-edge on node k mod N with
shift (2 r_max, 0, 0) -> length 2 r_max >= r_max -> PolynomialCutoff exactly 0 -> exactly-zero message and exactly-zero
position / displacement gradient.
Stress (stress=True): MACE recomputes shifts = unit_shifts @ (cell + cell@sym(displacement)) inside the model, so the
padded edges get unit_shift (m,0,0) with m = max(1, ceil(2 r_max / |cell[0]|)) computed ON THE GPU at every load
(no host sync) -> |shift| >= 2 r_max for the actual cell. MACE creates the displacement tensor itself inside the
captured region (fixed shape [1,3,3]); autograd.grad w.r.t. (positions, displacement) is captured with the forward.
Stress is returned in Voigt order (xx,yy,zz,yz,xz,xy) in eV/A^3, = MACECalculator results["stress"].

API (results/phase7/INTERFACES.md):
  build_backend(kind in {"e3nn","flash","hybrid"}, transforms=()) -> (calc, model)
  GraphedRunner(calc, model, *, edge_margin=0.20, mult=64, stress=False, builder="proto"|"stock", max_graphs=4)
  runner.step(atoms) -> dict(energy, forces[N,3], stress[6]|None, recaptured, E, E_cap)
  runner.step_tensors(atoms) -> same but torch views into graph outputs (no D2H; overwritten by the next replay)
  runner.stats() -> dict(captures, recaptures, last_capture_s, key, n_graphs, ...)
Import: put $R/scripts/graphed7 on sys.path and `import graphed_runner` BEFORE torch/mace imports elsewhere
(it imports scripts/cudagraph6/hybrid_gc first, which pins flashTP_e3nn to src/flashTP-graphcap).
"""
import os
import sys
import math
import time
import hashlib
from collections import OrderedDict

R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
for _p in (f"{R}/scripts/cudagraph6", f"{R}/scripts/graph"):
    if _p not in sys.path:
        sys.path.append(_p)
import hybrid_gc as HB  # noqa: E402  (first: pins flashTP_e3nn to src/flashTP-graphcap, asserts fp64 CG)

assert HB.flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-graphcap/"), HB.flashTP_e3nn.__file__
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mace.tools import torch_tools  # noqa: E402
from proto_graph import ProtoBuilder  # noqa: E402

EDGE_KEYS = ("edge_index", "shifts", "unit_shifts")
COPY_KEYS = ("positions", "cell", "rcell", "volume")
_KINDS = {"e3nn": ("e3nn", "e3nn"), "flash": ("flashtp", "e3nn"), "hybrid": ("flashtp", "cueq")}


def round_cap(E, frac=0.20, mult=64):
    return max(mult, int(math.ceil(E * (1 + frac) / mult)) * mult)


def build_backend(kind, transforms=(), dtype=torch.float64, model_path=HB.MODEL):
    """kind in {"e3nn","flash","hybrid"}; transforms: callables model -> handle, applied in order after the backend
    adapters are installed and before any capture. Handles are kept in calc.graphed7_transform_handles."""
    if kind not in _KINDS:
        raise ValueError(f"kind={kind!r} not in {sorted(_KINDS)}")
    conv, sc = _KINDS[kind]
    calc, info = HB.build_calc(model_path=model_path, conv=conv, sc=sc, dtype=dtype, check_indices=False)
    model = calc.models[0]
    handles = []
    for t in transforms:
        handles.append(t(model))
    calc.graphed7_kind = kind
    calc.graphed7_info = info
    calc.graphed7_transform_handles = handles
    return calc, model


def _stock_build(calc, atoms):
    b = calc._clone_batch(calc._atoms_to_batch(atoms))
    dt = next(calc.models[0].parameters()).dtype
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v):
            b[k] = v.to(dtype=dt)
    return b.to_dict()


class _Graph:
    """One captured graph for one key at fixed edge capacity E_cap. Owns all its buffers."""

    def __init__(self, model, bd, E_cap, r_max, stress, numbers, n_warm=3):
        t0 = time.perf_counter()
        self.model = model
        self.r_max = float(r_max)
        self.stress = bool(stress)
        self.numbers = np.array(numbers, copy=True)
        self.N = int(bd["positions"].shape[0])
        if self.N != len(self.numbers) or bd["node_attrs"].shape[0] != self.N:
            raise ValueError(f"batch N={self.N} / node_attrs {tuple(bd['node_attrs'].shape)} != len(numbers) "
                             f"{len(self.numbers)}")
        self.E_cap = int(E_cap)
        dev = bd["positions"].device
        st, self.src = {}, {}
        for k, v in bd.items():
            if k in EDGE_KEYS:
                continue
            st[k] = v.detach().clone()
            if k not in COPY_KEYS:
                self.src[k] = v          # identity of the static source tensors captured from
        st["positions"].requires_grad_(False)
        N = self.N
        ar = torch.arange(self.E_cap, device=dev, dtype=bd["edge_index"].dtype)
        node = (ar % N) if N > 0 else ar * 0
        self.pad_ei = torch.stack((node, node)).contiguous()
        self.pad_sh = torch.zeros((self.E_cap, 3), dtype=bd["shifts"].dtype, device=dev)
        self.pad_sh[:, 0] = 2.0 * self.r_max
        st["edge_index"] = self.pad_ei.clone()
        st["shifts"] = self.pad_sh.clone()
        st["unit_shifts"] = torch.zeros((self.E_cap, 3), dtype=bd["unit_shifts"].dtype, device=dev)
        self.static = st
        self.load(bd)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._run()
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - t0

    def _run(self):
        # shallow copy: prepare_graph() reassigns data["positions"]/data["shifts"] when stress is on; the static
        # input buffers must stay the tensors we copy into.
        d = dict(self.static)
        with torch_tools.default_dtype("float64" if d["positions"].dtype == torch.float64 else "float32"):
            o = self.model(d, compute_force=True, compute_stress=self.stress, training=False)
        return dict(energy=o["energy"], forces=o["forces"], stress=o["stress"] if self.stress else None)

    @torch.no_grad()
    def load(self, bd):
        st = self.static
        E = int(bd["edge_index"].shape[1])
        if E > self.E_cap:
            raise RuntimeError(f"E={E} > E_cap={self.E_cap} (caller must recapture)")
        if bd["positions"].shape[0] != self.N:
            raise RuntimeError(f"N mismatch {bd['positions'].shape[0]} != {self.N}")
        for k, v in self.src.items():   # per-node / per-config statics: refresh if a different tensor is passed
            w = bd[k]
            if w is not v:
                if w.shape != st[k].shape or w.dtype != st[k].dtype:
                    raise RuntimeError(f"static {k}: {tuple(w.shape)}/{w.dtype} != captured "
                                       f"{tuple(st[k].shape)}/{st[k].dtype}")
                st[k].copy_(w)
                self.src[k] = w
        for k in COPY_KEYS:
            st[k].copy_(bd[k].detach())
        st["edge_index"][:, :E].copy_(bd["edge_index"])
        st["edge_index"][:, E:].copy_(self.pad_ei[:, E:])
        st["shifts"][:E].copy_(bd["shifts"])
        st["shifts"][E:].copy_(self.pad_sh[E:])
        st["unit_shifts"][:E].copy_(bd["unit_shifts"])
        st["unit_shifts"][E:].zero_()
        if self.stress and E < self.E_cap:
            # shifts are recomputed in-model from unit_shifts @ cell: pick m with m*|a1| >= 2 r_max (GPU, no sync)
            a1 = torch.linalg.vector_norm(st["cell"].view(-1, 3, 3)[0, 0])
            m = torch.clamp(torch.ceil(2.0 * self.r_max / a1), min=1.0)
            st["unit_shifts"][E:, 0].copy_(m.expand(self.E_cap - E))
        return E

    def replay(self):
        self.graph.replay()
        return self.out


class GraphedRunner:
    def __init__(self, calc, model, *, edge_margin=0.20, mult=64, stress=False, builder="proto", max_graphs=4,
                 recapture_margin=0.25):
        if model is not calc.models[0]:
            raise ValueError("model must be calc.models[0]")
        self.calc, self.model = calc, model
        self.edge_margin, self.mult, self.recapture_margin = float(edge_margin), int(mult), float(recapture_margin)
        self.stress = bool(stress)
        self.r_max = float(calc.r_max)
        self.dtype = next(model.parameters()).dtype
        self.max_graphs = int(max_graphs)
        if builder == "proto":
            self._pb = ProtoBuilder(calc)
            self._build = self._pb.build
        elif builder == "stock":
            self._build = lambda a: _stock_build(calc, a)
        else:
            raise ValueError(f"builder={builder!r}")
        self.builder = builder
        self.graphs = OrderedDict()
        self._captures = 0
        self._recaptures = 0
        self._evictions = 0
        self._last_capture_s = 0.0
        self._last_key = None
        self.events = []

    def key_of(self, atoms):
        pbc = tuple(bool(x) for x in atoms.get_pbc())
        cell_none = not np.asarray(atoms.cell).any()
        return (len(atoms), atoms.numbers.tobytes(), pbc, cell_none, self.calc.head, str(self.dtype), self.stress)

    @staticmethod
    def describe_key(key):
        n, nb, pbc, cn, head, dt, st = key
        return dict(n_atoms=n, numbers_sha1=hashlib.sha1(nb).hexdigest()[:12], pbc=pbc, cell_is_none=cn, head=head,
                    dtype=dt, stress=st)

    def _capture(self, key, bd, atoms, cap, reason):
        old = self.graphs.pop(key, None)
        old_cap = old.E_cap if old is not None else None
        del old
        while len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
            self._evictions += 1
        g = _Graph(self.model, bd, cap, self.r_max, self.stress, atoms.numbers)
        self.graphs[key] = g
        self._captures += 1
        self._last_capture_s = g.capture_s
        self.events.append(dict(reason=reason, key=self.describe_key(key), old_cap=old_cap, E_cap=cap,
                                E=int(bd["edge_index"].shape[1]), capture_s=g.capture_s))
        return g

    def _step_bd(self, key, atoms, bd):
        E = int(bd["edge_index"].shape[1])
        g = self.graphs.get(key)
        recaptured = False
        if g is None:
            g = self._capture(key, bd, atoms, round_cap(E, self.edge_margin, self.mult), "new_key")
        elif E > g.E_cap:
            g = self._capture(key, bd, atoms, round_cap(E, self.recapture_margin, self.mult), "edge_overflow")
            self._recaptures += 1
            recaptured = True
        else:
            if not np.array_equal(g.numbers, atoms.numbers):  # redundant with the key; cheap host check
                raise RuntimeError("species mismatch for cached graph (key collision?)")
            self.graphs.move_to_end(key)
            g.load(bd)
        self._last_key = key
        out = g.replay()
        return out, recaptured, E, g.E_cap

    def step_tensors(self, atoms):
        key = self.key_of(atoms)
        bd = self._build(atoms)
        out, rec, E, cap = self._step_bd(key, atoms, bd)
        return dict(energy=out["energy"], forces=out["forces"], stress=out["stress"], recaptured=rec, E=E, E_cap=cap)

    def step(self, atoms):
        o = self.step_tensors(atoms)
        st = None
        if o["stress"] is not None:
            s = o["stress"].detach()[0].cpu().numpy()
            st = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        return dict(energy=float(o["energy"].detach()[0].item()), forces=o["forces"].detach().cpu().numpy().copy(),
                    stress=st, recaptured=o["recaptured"], E=o["E"], E_cap=o["E_cap"])

    def stats(self):
        return dict(captures=self._captures, recaptures=self._recaptures, evictions=self._evictions,
                    last_capture_s=self._last_capture_s,
                    key=self.describe_key(self._last_key) if self._last_key is not None else None,
                    n_graphs=len(self.graphs), E_caps=[g.E_cap for g in self.graphs.values()])
