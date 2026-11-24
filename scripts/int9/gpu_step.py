"""int9 GPUStep: GPU-resident MACE step (neighbour list + model fwd + force [+ stress] backward) captured as ONE CUDA graph.

  sim = GPUStep(calc, model, atoms_template, stress=False, nl="auto")   # model built with common9.build("nodeext+embed+node")
  E, F = sim.forces(pos_gpu)            # (stress=True: E, F, S)  GPU tensors = graph outputs (overwritten by next call)
  rep = sim.check()                     # explicit host sync on the flag copy; recaptures on overflow (see policy)
  sim.stats()                           # captures, recaptures, checks, syncs, E_cap, last n_edges, path, ...
  sim.set_atoms(atoms)                  # switch template (species order / N / cell / pbc) -> keyed graph (LRU, max_graphs)

Step graph (per key = (N, numbers bytes, pbc, cell bytes, dtype, stress, head)) at fixed (E_cap, NL capacity):
  NL.build(static pos) -> dst-sorted edge_index/unit_shifts/shifts + rowptr_dst + perm_src/rowptr_src + n_edges/overflow
  -> CSR fix-up (rowptr_*[N] := E_cap, perm_src[k] := k for k >= n_edges: padding src=dst=N-1 is covered by both CSRs)
  -> nodeconv_ext.ExtCsr.set(...) (perm_dst = arange(E_cap): edges are already dst-sorted) -> NO sort inside the model
  -> model(batch, compute_force=True, compute_stress) -> energy/forces/stress poisoned with NaN if overflow
  -> sticky device flags (overflow_any, max n_edges) and a D2H copy of [overflow_any, max_E, last_E] into pinned memory
     (all inside the graph; zero host syncs per step).
NL implementation: nl="cell" -> scripts/nl9/cell_nl.CellNeighborList (agent A); "brute" -> scripts/int9/brute_nl stub
  (O(N^2), capturable, slow: explicit nl="brute" only); "auto" -> cell (if results/nl9/READY); if the cell list raises
  NotImplementedError (periodic box with < 3 bins per dim, e.g. wbox192 L=12.4 A; mixed pbc) -> "proto" fallback path: ProtoGPU (O(N^2) + nonzero host sync) outside the graph +
  graphed model (graphed7 _Graph, in-model CSR sort).
Overflow policy (n_edges > E_cap or NL bin overflow):
  * results of an overflowed step are NaN (energy, forces, stress) -> can never be mistaken for valid values;
  * the flag is sticky on the device until check(); check() waits for the last step's flag copy (1 event sync, counted),
    and if any step since the previous check overflowed it recaptures with E_cap = round_cap(max true n_edges, 0.25)
    (and doubled NL bin capacity if the builder reports a bin overflow) and returns ok=False: the caller must discard
    every result since its last ok check (roll back to the state saved then) and recompute;
  * proactive: if max n_edges since the last check > (1 - headroom) * E_cap, check() recaptures with a larger E_cap while
    results are still valid (ok=True, recaptured=True);
  * forces_checked(pos) = forces + check (+ recompute on the new graph if needed): always valid, 1 sync per call.
"""
import gc
import os
import sys
import time
import math
from collections import OrderedDict
R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
for _p in (f"{R}/scripts/int9", f"{R}/scripts/nl9"):
    if _p not in sys.path:
        sys.path.append(_p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mace.tools import torch_tools  # noqa: E402
import graphed_runner as GR  # noqa: E402
import nodeconv_ext as NX  # noqa: E402
from proto_graph import ProtoBuilder  # noqa: E402
from brute_nl import BruteNeighborList  # noqa: E402

EDGE_KEYS = ("edge_index", "shifts", "unit_shifts")


def round_cap(E, frac, mult=64):
    return max(mult, int(math.ceil(E * (1 + frac) / mult)) * mult)


BRUTE_MAX_PAIRS = 30_000_000   # auto: brute stub only when N^2 * images <= this (small periodic boxes < 3 bins)


def _nl_chain(nl):
    """ordered list of (class, name) to try; NotImplementedError from a constructor -> next; empty -> proto fallback."""
    if nl == "brute":
        return [(BruteNeighborList, "brute")]
    if nl == "proto":
        return []
    chain = []
    if nl == "cell" or (nl == "auto" and os.path.exists(f"{R}/results/nl9/READY")):
        import cell_nl
        chain.append((cell_nl.CellNeighborList, "cell"))
    # auto: NO brute stub fallback -- measured (final_run1, wbox192): whole-graph brute 6.05 ms vs proto path 2.38 ms
    if not chain:
        raise ValueError(nl)
    return chain


class _StepGraph:
    """One whole-step graph for one key at fixed E_cap / NL capacity."""

    def __init__(self, owner, atoms, E_cap, nl_kw):
        t0 = time.perf_counter()
        o = owner
        self.o, self.N, self.E_cap, self.nl_kw = o, len(atoms), int(E_cap), dict(nl_kw)
        self.numbers = np.array(atoms.numbers, copy=True)
        dev = o.dev
        tmpl = o.pb._template(atoms)
        self.keys = tmpl["keys"]
        self.static = {k: v for k, v in tmpl["static"].items()}
        # cell/rcell/volume (constant per key): stock values via ProtoGPU logic
        from common9 import ProtoGPU
        pg = ProtoGPU(o.calc, atoms)
        self.cell, self.rcell, self.vol = pg.cell.clone(), pg.rcell.clone(), pg.vol.clone()
        self.NL = o.nl_cls(o.r_max, tuple(bool(x) for x in atoms.pbc), np.asarray(atoms.get_cell()), self.N, self.E_cap,
                           device=dev, **self.nl_kw)
        self.pos = torch.tensor(atoms.positions, dtype=torch.float64, device=dev)
        self.perm_dst = torch.arange(self.E_cap, dtype=torch.int64, device=dev)
        self.flags = torch.zeros(3, dtype=torch.int64, device=dev)       # [overflow_any, max_E, last_E] (sticky)
        self.host = torch.zeros(3, dtype=torch.int64).pin_memory()
        self.ext = NX.get_ext(o.model)
        self.c_nan = torch.full((), float("nan"), dtype=torch.float64, device=dev)
        self.c_zero = torch.zeros((), dtype=torch.float64, device=dev)
        self.cap_t = torch.full((1,), self.E_cap, dtype=torch.int32, device=dev)
        self.node_ar = torch.arange(self.N, dtype=torch.int64, device=dev)
        self.ar_cap = torch.arange(self.E_cap, dtype=torch.int64, device=dev)
        self.zero1 = torch.zeros(1, dtype=torch.int64, device=dev)
        # Python's cyclic GC must not run inside the capture: freeing an old graph's pinned flag buffer makes the
        # caching host allocator record a CUDA event -> capture invalidated (seen as an abort in test9 after many
        # GPUStep objects were dropped). Collect first, disable during warmup + capture.
        gc.collect(); torch.cuda.synchronize()
        was = gc.isenabled(); gc.disable()
        try:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._run()
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            self.flags.zero_()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._run()
            torch.cuda.synchronize()
        finally:
            if was:
                gc.enable()
        self.ext.clear() if self.ext is not None else None
        self.event = torch.cuda.Event()
        self.capture_s = time.perf_counter() - t0

    def _run(self):
        o, N, Ec = self.o, self.N, self.E_cap
        nl = self.NL.build(self.pos)
        ei = nl["edge_index"]
        ne = nl["n_edges"].reshape(()).to(torch.int64)
        obits = nl["overflow"].reshape(()).to(torch.int64)          # nl9: bits {1: E>E_cap, 2: bin, 4: reverse}; stub: 0/1
        ovf = obits != 0
        if o.pad_mode == "spread":
            ei, pd, rp_d, ps, rp_s = self._spread_padding(nl, ne, ovf)
        elif "rowptr_dst_pad" in nl:   # nl9 emits the padded CSR (== build_csr on the padded edge_index), identity perm_src on padding
            pd, rp_d, rp_s, ps = self.perm_dst, nl["rowptr_dst_pad"], nl["rowptr_src_pad"], nl["perm_src"]
        else:  # CSR fix-up: padding (src=dst=N-1, slots [E, E_cap)) covered by both CSRs
            pd = self.perm_dst
            rp_d = torch.cat((nl["rowptr_dst"][:N].to(torch.int32), self.cap_t))
            rp_s = torch.cat((nl["rowptr_src"][:N].to(torch.int32), self.cap_t))
            ps = torch.where(self.perm_dst >= torch.clamp(ne, max=Ec), self.perm_dst, nl["perm_src"].to(torch.int64))
        if self.ext is not None:
            self.ext.set(ei, N, pd, rp_d, ps, rp_s)
        d = dict(self.static)
        d.update(positions=self.pos, edge_index=ei, shifts=nl["shifts"], unit_shifts=nl["unit_shifts"],
                 cell=nl.get("cell", self.cell), rcell=nl.get("rcell", self.rcell), volume=nl.get("volume", self.vol))
        d = {k: d[k] for k in self.keys}
        with torch_tools.default_dtype("float64"):
            out = o.model(d, compute_force=True, compute_stress=o.stress, training=False)
        nan = torch.where(ovf, self.c_nan, self.c_zero)
        res = dict(energy=out["energy"] + nan, forces=out["forces"] + nan,
                   stress=(out["stress"] + nan) if o.stress else None)
        # sticky flags + async D2H into pinned memory (captured memcpy)
        self.flags[0].copy_(torch.bitwise_or(self.flags[0], obits))
        self.flags[1].copy_(torch.maximum(self.flags[1], ne))
        self.flags[2].copy_(ne)
        self.host.copy_(self.flags, non_blocking=True)
        self.nl_out = nl
        return res

    def _spread_padding(self, nl, ne, ovf):
        """Re-home the P = E_cap - E padding self-edges from node N-1 onto nodes k mod N (k = 0..P-1), as the graphed7
        runner does, and build the matching CSRs WITHOUT a sort (O(E_cap) elementwise + one scatter per order).
        Reason: with all padding on node N-1 the node-centric conv gives one CTA ~P edges (fwd: dst row, bwd: src row)
        and MACE's index backward accumulates P duplicates -> a long serial tail (measured: 10-40x slower model step at
        2187 atoms). Edge slots are unchanged (real edges [0,E), padding [E,E_cap)); only padding edge_index values and
        the CSR permutations / row pointers change.
          padcnt(n) = #{k < P : k mod N == n};  cumpad = exclusive cumsum;  rowptr'(n) = rowptr(n) + cumpad(n)
          dst order: real edge e (dst d) -> slot e + cumpad[d]; padding k -> rowptr'(n) + deg(n) + k // N
          src order: real edge at src-sorted position t (src s) -> slot t + cumpad[s]; padding k -> same as dst
        """
        N, Ec = self.N, self.E_cap
        # on ANY overflow the builder's edges/CSR may be inconsistent (bin overflow: missing reverse edges, invalid
        # perm_src) -> feeding them to the node conv caused an illegal memory access (context lost). Treat every slot as
        # padding instead: the model then sees a valid all-padding graph; results are NaN-poisoned anyway.
        Er = torch.where(ovf, torch.zeros_like(ne), torch.clamp(ne, max=Ec))
        P = Ec - Er
        n = self.node_ar                                            # [N] int64
        padcnt = torch.clamp(torch.div(P - n + (N - 1), N, rounding_mode="floor"), min=0)
        cumpad = torch.cat((self.zero1, torch.cumsum(padcnt, 0)))  # [N+1]
        rp = torch.cat((nl["rowptr_dst"][:N].to(torch.int64), Er.reshape(1)))   # real CSR (symmetric list: dst == src)
        rp = torch.where(ovf, torch.zeros_like(rp), rp)             # overflow: all-padding graph (see above)
        deg = rp[1:] - rp[:-1]
        rpn = rp + cumpad                                           # rowptr' [N+1], rpn[N] = E_cap
        ids = self.ar_cap
        is_pad = ids >= Er
        k = torch.clamp(ids - Er, min=0)
        pnode = torch.remainder(k, N)
        ei0 = nl["edge_index"]
        ei = torch.where(is_pad.unsqueeze(0), pnode.unsqueeze(0).expand(2, Ec), ei0)
        pad_slot = rpn[pnode] + deg[pnode] + torch.div(k, N, rounding_mode="floor")
        dnode = ei0[1]
        slot_d = torch.where(is_pad, pad_slot, ids + cumpad[dnode.clamp(0, N - 1)])
        perm_d = torch.empty_like(ids).scatter_(0, slot_d, ids)
        ps0 = nl["perm_src"].to(torch.int64)                        # src-sorted position t -> edge (real part)
        e_t = torch.where(is_pad, ids, ps0)
        snode = ei0[0][e_t.clamp(0, Ec - 1)]
        slot_s = torch.where(is_pad, pad_slot, ids + cumpad[snode.clamp(0, N - 1)])
        perm_s = torch.empty_like(ids).scatter_(0, slot_s, e_t)
        rpn32 = rpn.to(torch.int32)
        return ei, perm_d, rpn32, perm_s, rpn32

    def replay(self):
        self.graph.replay()
        self.event.record()
        return self.out


class GPUStep:
    def __init__(self, calc, model, atoms, *, stress=False, nl="auto", edge_margin=0.20, recapture_margin=0.25,
                 headroom=0.02, max_graphs=4, nl_kw=None, E_cap=None, pad_mode="spread"):
        if model is not calc.models[0]:
            raise ValueError("model must be calc.models[0]")
        self.calc, self.model = calc, model
        self.pad_mode = pad_mode            # "spread" (default) | "last" (builder's padding on node N-1; slow, for comparison)
        self.dev = torch.device("cuda")
        self.r_max = float(calc.r_max)
        self.stress = bool(stress)
        self.dtype = next(model.parameters()).dtype
        if self.dtype != torch.float64:
            raise ValueError("fp64 only")
        self.edge_margin, self.recapture_margin, self.headroom = edge_margin, recapture_margin, headroom
        self.max_graphs = max_graphs
        self.nl_req = nl
        self.nl_kw = dict(nl_kw or {})
        self.pb = ProtoBuilder(calc)
        self.graphs = OrderedDict()
        self.fallback = {}
        self.n_captures = self.n_recaptures = self.n_checks = self.n_syncs = self.n_overflows = self.n_steps = 0
        self.n_proactive = self.n_evictions = 0
        self.last_capture_s = 0.0
        self.events = []
        self._E_cap_req = E_cap
        self.set_atoms(atoms)

    # ---------------------------------------------------------------- keys / capture
    def key_of(self, atoms):
        return (len(atoms), atoms.numbers.tobytes(), tuple(bool(x) for x in atoms.pbc),
                np.asarray(atoms.get_cell(), dtype=np.float64).tobytes(), str(self.dtype), self.stress, self.calc.head)

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        self.key = self.key_of(atoms)
        self.pb._check_supported(atoms)
        if self.stress and not all(atoms.pbc):
            raise ValueError("stress needs a fully periodic cell")
        if self.key in self.graphs:
            self.graphs.move_to_end(self.key); self.path = "graph"; return
        if self.key in self.fallback:
            self.path = "proto"; return
        from common9 import ProtoGPU
        pos = torch.tensor(atoms.positions, dtype=torch.float64, device=self.dev)
        E0 = None
        for cls, name in _nl_chain(self.nl_req):
            if name == "brute" and self.nl_req == "auto":
                M = 27 if all(atoms.pbc) else 1
                if len(atoms) ** 2 * M > BRUTE_MAX_PAIRS:
                    continue
            try:
                self.nl_cls, self.nl_name = cls, name
                if E0 is None:   # initial E from the proto builder (setup only; one sync)
                    E0 = int(ProtoGPU(self.calc, atoms).build(pos)["edge_index"].shape[1])
                cap = self._E_cap_req or round_cap(E0, self.edge_margin)
                self._capture(atoms, cap, dict(self.nl_kw), "new_key", E0)
                self.path = "graph"
                return
            except NotImplementedError as ex:
                self.events.append(dict(reason=f"nl_{name}_unsupported", err=repr(ex)[:200]))
        self.events.append(dict(reason="fallback_proto"))
        self._make_fallback(atoms)
        self.path = "proto"

    def _capture(self, atoms, cap, nl_kw, reason, E):
        old = self.graphs.pop(self.key, None)
        old_cap = old.E_cap if old is not None else None
        del old
        torch.cuda.synchronize()
        while len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False); self.n_evictions += 1
        g = _StepGraph(self, atoms, cap, nl_kw)
        self.graphs[self.key] = g
        self.n_captures += 1
        self.last_capture_s = g.capture_s
        self.events.append(dict(reason=reason, N=len(atoms), E=int(E), old_cap=old_cap, E_cap=cap, nl=self.nl_name,
                                capture_s=g.capture_s))
        return g

    def _make_fallback(self, atoms):
        from common9 import ProtoGPU
        self.nl_name = "proto"
        try:
            pg = ProtoGPU(self.calc, atoms)
        except NotImplementedError:      # mixed pbc: host proto_graph (positions D2H each step)
            pg = None
            self.nl_name = "proto_host"
        self.fallback[self.key] = dict(pg=pg, runner=GR.GraphedRunner(self.calc, self.model, stress=self.stress),
                                       atoms=atoms.copy())

    # ---------------------------------------------------------------- step
    def forces(self, pos=None):
        self.n_steps += 1
        if self.path == "proto":
            fb = self.fallback[self.key]
            p = pos if pos is not None else torch.tensor(fb["atoms"].positions, dtype=torch.float64, device=self.dev)
            rn = fb["runner"]
            if fb["pg"] is None:
                fb["atoms"].positions = p.detach().cpu().numpy()   # D2H (sync)
                bd = rn._build(fb["atoms"])
            else:
                bd = fb["pg"].build(p)                              # nonzero() -> host sync (counted)
            self.n_syncs += 1
            out, rec, E, cap = rn._step_bd(rn.key_of(fb["atoms"]), fb["atoms"], bd)
            self._fb_last = (E, cap)
            res = (out["energy"], out["forces"])
            return res + (out["stress"][0],) if self.stress else res
        g = self.graphs[self.key]
        if pos is not None and pos is not g.pos:
            with torch.no_grad():
                g.pos.copy_(pos)
        out = g.replay()
        res = (out["energy"], out["forces"])
        return res + (out["stress"][0],) if self.stress else res

    def replay(self):
        """graph replay only (positions already in sim.positions); proto path: replay of the model graph with the last
        loaded neighbour list (no NL build)."""
        if self.path == "proto":
            fb = self.fallback[self.key]; rn = fb["runner"]
            return rn.graphs[rn.key_of(fb["atoms"])].replay()
        return self.graphs[self.key].replay()

    @property
    def positions(self):
        """static position buffer of the current graph (write in place under torch.no_grad() to skip the D2D copy)."""
        return self.graphs[self.key].pos if self.path == "graph" else None

    def check(self):
        self.n_checks += 1
        if self.path == "proto":
            return dict(ok=True, recaptured=False, path="proto")
        g = self.graphs[self.key]
        g.event.synchronize(); self.n_syncs += 1
        ovf, maxE, lastE = (int(x) for x in g.host.tolist())
        g.flags.zero_()
        rep = dict(ok=not ovf, overflow=ovf, max_E=maxE, last_E=lastE, E_cap=g.E_cap, recaptured=False)
        bin_ovf = bool(ovf & 6)
        if ovf:
            self.n_overflows += 1
        if ovf or maxE > (1.0 - self.headroom) * g.E_cap:
            kw = dict(g.nl_kw)
            if bin_ovf:   # nl9 bin overflow: enlarge bin capacity (max_bin of the LAST step + 50%, at least 2x)
                mb = int(g.nl_out["max_bin"].item()) if "max_bin" in g.nl_out else 0
                old = int(getattr(g.NL, "bin_cap", 64))
                kw["bin_cap"] = int(max(2 * old, math.ceil(1.5 * mb / 32) * 32))
                rep["new_bin_cap"] = kw["bin_cap"]
            # on a bin overflow n_edges may be undercounted -> keep at least the current cap
            cap = max(round_cap(maxE, self.recapture_margin), g.E_cap if bin_ovf else 0)
            pos_keep = g.pos.clone()
            g2 = self._capture(self.atoms, cap, kw, "overflow" if ovf else "proactive", maxE)
            with torch.no_grad():
                g2.pos.copy_(pos_keep)
            self.n_recaptures += 1
            if not ovf:
                self.n_proactive += 1
            rep.update(recaptured=True, new_E_cap=cap)
        return rep

    def forces_checked(self, pos):
        for _ in range(4):
            res = self.forces(pos)
            rep = self.check()
            if rep["ok"]:
                return res
        raise RuntimeError("overflow persists after 4 recaptures")

    def edges(self):
        """(edge_index [2,E], unit_shifts [E,3]) of the last step (host sync; testing only)."""
        g = self.graphs[self.key]
        torch.cuda.synchronize()
        E = int(g.nl_out["n_edges"].item())
        E = min(E, g.E_cap)
        return g.nl_out["edge_index"][:, :E], g.nl_out["unit_shifts"][:E]

    def stats(self):
        g = self.graphs.get(self.key) if self.path == "graph" else None
        last = int(g.host[2]) if g is not None else (self._fb_last[0] if hasattr(self, "_fb_last") else None)
        fbcap = self._fb_last[1] if (g is None and hasattr(self, "_fb_last")) else None
        return dict(path=self.path, nl=self.nl_name, captures=self.n_captures, recaptures=self.n_recaptures,
                    proactive_recaptures=self.n_proactive, overflows=self.n_overflows, checks=self.n_checks,
                    host_syncs=self.n_syncs, steps=self.n_steps, evictions=self.n_evictions,
                    last_capture_s=self.last_capture_s, E_cap=(g.E_cap if g is not None else fbcap), n_edges_last=last,
                    n_graphs=len(self.graphs), ext_csr_layer_calls=getattr(NX.get_ext(self.model), "n_ext", None),
                    sort_fallback_layer_calls=getattr(NX.get_ext(self.model), "n_sort", None), events=self.events[-10:])
