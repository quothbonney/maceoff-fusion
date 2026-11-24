"""CUDA-graph capture/replay of the whole MACE energy+forces call (forward + autograd.grad) at fixed edge capacity.

Padding design (documented, verified in run_cg6.py):
  * N fixed per system (static node tensors cloned from the first batch dict).
  * Edge arrays allocated at capacity E_cap >= E. Real edges occupy [0, E); padded edge k (k >= E) is a self-edge
    sender = receiver = k mod N (spread over nodes so the FlashTP scatter atomics do not all hit node 0) with
    shift = (2 r_max, 0, 0) and unit_shift = 0. Its length is 2 r_max, so MACE's PolynomialCutoff
    `envelope * (x < r_max)` is exactly 0, the radial features (Bessel * cutoff) are exactly 0, the radial MLP
    (no bias, silu(0)=0) gives exactly-zero TP weights, and the message added to the receiver is exactly +0.0. The SH input
    vector is non-zero (no NaN), and the position gradient of a self-edge is g - g = 0 exactly.
  * E = 0 needs no eager fallback: all E_cap edges are padding (the kernel always sees E_cap > 0).
  * E > E_cap -> recapture at ceil(1.25 E) (cost recorded in self.recaptures).
Requirements: model built with check_indices=False (no host sync), FlashTP from the graph-capture build (kernels and
constant-memory uploads on torch's current stream). Warmup runs on the side stream before capture (cueq JIT, TorchScript
profiling).
"""
import math
import time
import torch
from mace.tools import torch_tools

EDGE_KEYS = ("edge_index", "shifts", "unit_shifts")
COPY_KEYS = ("positions", "cell", "rcell", "volume")


def round_cap(E, frac=0.20, mult=64):
    return max(mult, int(math.ceil(E * (1 + frac) / mult)) * mult)


class GraphedMACE:
    def __init__(self, model, bd, E_cap, r_max, n_warm=3):
        self.model = model
        self.r_max = float(r_max)
        self.N = bd["positions"].shape[0]
        self.recaptures = []
        self._bd_template = bd
        self.capture_s = self._setup(bd, E_cap, n_warm)

    # ---------------------------------------------------------------------------------------------------------
    def _setup(self, bd, E_cap, n_warm):
        t0 = time.perf_counter()
        self.E_cap = int(E_cap)
        dev = bd["positions"].device
        st = {}
        for k, v in bd.items():
            if k in EDGE_KEYS:
                continue
            st[k] = v.detach().clone()
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
        self._load(bd)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out_E, self.out_F = self._run()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _run(self):
        with torch_tools.default_dtype("float64"):
            out = self.model(self.static, compute_force=True, training=False)
        return out["energy"], out["forces"]

    def _load(self, bd):
        with torch.no_grad():
            self._load_ng(bd)

    def _load_ng(self, bd):
        st = self.static
        E = bd["edge_index"].shape[1]
        for k in COPY_KEYS:
            st[k].copy_(bd[k].detach())
        st["edge_index"][:, :E].copy_(bd["edge_index"])
        st["edge_index"][:, E:].copy_(self.pad_ei[:, E:])
        st["shifts"][:E].copy_(bd["shifts"])
        st["shifts"][E:].copy_(self.pad_sh[E:])
        st["unit_shifts"][:E].copy_(bd["unit_shifts"])
        st["unit_shifts"][E:].zero_()

    # ---------------------------------------------------------------------------------------------------------
    def step(self, bd):
        """Copy a (unpadded) batch dict into the static buffers and replay. Returns views of the static outputs
        (overwritten by the next step; clone if you keep them)."""
        E = bd["edge_index"].shape[1]
        if E > self.E_cap:
            old = self.E_cap
            del self.graph, self.out_E, self.out_F
            dt = self._setup(bd, round_cap(E, 0.25), 3)
            self.recaptures.append(dict(old_cap=old, new_cap=self.E_cap, E=E, capture_s=dt))
        else:
            self._load(bd)
        self.graph.replay()
        return self.out_E, self.out_F

    def replay(self):
        self.graph.replay()
        return self.out_E, self.out_F
