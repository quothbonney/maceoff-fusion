"""fuse7_node: fused post-convolution node linear kernels for MACE-OFF23 (MACE 0.3.17), fp64, Triton 3.6.

What is replaced (in-memory, per interaction block, via an instance-level `forward` override; class/sources untouched):
  L0 RealAgnosticInteractionBlock:   reshape(skip_tp(linear(m) / avg, node_attrs))  ->  ONE kernel
        folded per-species weight Wf[s,l] = (linear . /avg . skip_tp(., onehot_s)) restricted to irrep l, written directly in
        the reshape_irreps layout [N, 128, 16].
  L1 RealAgnosticResidualInteractionBlock:
        reshape(linear(m) / avg)        ->  ONE kernel (1/avg folded into the 1280->128 per-l weights, reshape layout out)
        sc = skip_tp(node_feats, attrs) ->  ONE kernel (per-species 128x128 on the 0e part; 1o part provably unused)
Folded weights are extracted by evaluating the ORIGINAL modules on identity bases (so every e3nn normalisation constant is
taken exactly as the stock code applies it), and the dense map is verified to have exactly the block structure
(same-l, same-m, m-independent) the kernel assumes; install() raises if any entry deviates by > 1e-14 relative.

General (non one-hot) node_attrs are supported exactly: out[n] = sum_s attrs[n,s] * (W_s x[n]); species with an all-zero
attrs column inside a row tile are skipped (one-hot => only species present in the tile are computed; 1.0*acc is exact).
Backward: grad wrt x only (weights and node_attrs are constants; forward raises if node_attrs.requires_grad);
double backward raises (once_differentiable). Saved tensors: node_attrs only (the op is linear in x).
CUDA-graph capturable: no host syncs, grid depends only on N, launches on torch's current stream (Triton default).

API: install(model) -> handle; uninstall(model, handle=None); describe() -> dict.
"""
import hashlib
import inspect
import types

import torch
import triton
import triton.language as tl
from e3nn import o3

_STATE = "_fuse7_node_state"
_OPS = "_fuse7_node_ops"
# tile choice from results/fuse7_node/tune.json (job 1037, graph replay, H100): BLOCK_R=16 for small row counts, else 32.
BLOCK_R, BLOCK_U, BLOCK_K, NUM_WARPS = None, 64, 32, 4  # BLOCK_R=None -> heuristic below; set an int to force


def _block_r(rows):
    if BLOCK_R is not None:
        return BLOCK_R
    return 16 if rows <= 1024 else 32
STRUCT_TOL = 1e-14


class FuseNodeError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------- kernel
@triton.jit
def _seg_linear_kernel(X, x_rs, Y, y_rs, A, a_rs, S, W, SEG, INB, OUTB, N,
                       HAS_A: tl.constexpr, BR: tl.constexpr, BU: tl.constexpr, BK: tl.constexpr):
    """Y[n, outb[u] + m] = sum_s A[n,s] sum_k W[woff + (s*K + k)*U + u] * X[n, inb[k] + m],  rows r=(n,m), m < D.
    SEG row (int64 x 6): D, K, U, inb_off, outb_off, w_off."""
    pid_r = tl.program_id(0)
    seg = tl.program_id(1)
    pid_u = tl.program_id(2)
    D = tl.load(SEG + seg * 6 + 0)
    K = tl.load(SEG + seg * 6 + 1)
    U = tl.load(SEG + seg * 6 + 2)
    inb_off = tl.load(SEG + seg * 6 + 3)
    outb_off = tl.load(SEG + seg * 6 + 4)
    w_off = tl.load(SEG + seg * 6 + 5)
    r0 = pid_r * BR
    u0 = pid_u * BU
    if (r0 < N * D) & (u0 < U):
        rows = r0 + tl.arange(0, BR)
        rmask = rows < N * D
        n = rows // D
        m = rows - n * D
        us = u0 + tl.arange(0, BU)
        umask = us < U
        outb = tl.load(OUTB + outb_off + us, mask=umask, other=0)
        total = tl.zeros((BR, BU), dtype=tl.float64)
        for s in range(0, S):
            if HAS_A:
                a = tl.load(A + n * a_rs + s, mask=rmask, other=0.0)
                live = tl.max(tl.abs(a), axis=0) > 0.0
            else:
                a = tl.full((BR,), 1.0, tl.float64)
                live = True
            if live:
                acc = tl.zeros((BR, BU), dtype=tl.float64)
                for k0 in range(0, K, BK):
                    ks = k0 + tl.arange(0, BK)
                    kmask = ks < K
                    inb = tl.load(INB + inb_off + ks, mask=kmask, other=0)
                    xa = tl.load(X + n[:, None] * x_rs + inb[None, :] + m[:, None],
                                 mask=rmask[:, None] & kmask[None, :], other=0.0)
                    w = tl.load(W + w_off + (s * K + ks[:, None]) * U + us[None, :],
                                mask=kmask[:, None] & umask[None, :], other=0.0)
                    acc += tl.dot(xa, w)
                total += a[:, None] * acc
        tl.store(Y + n[:, None] * y_rs + outb[None, :] + m[:, None], total,
                 mask=rmask[:, None] & umask[None, :])


class SegPlan:
    """Device tables for one direction (fwd or bwd) of a segmented (optionally species-gathered) linear map."""

    def __init__(self, segs, in_dim, out_dim, out_shape_tail, n_species, device, zero_fill):
        # segs: list of dict(D, inb[list K], outb[list U], W [S,K,U] float64 tensor)
        rows, inb_all, outb_all, w_all = [], [], [], []
        io = oo = wo = 0
        for sg in segs:
            K, U = len(sg["inb"]), len(sg["outb"])
            assert tuple(sg["W"].shape) == (n_species, K, U), (sg["W"].shape, n_species, K, U)
            rows.append([sg["D"], K, U, io, oo, wo])
            inb_all += list(sg["inb"]); outb_all += list(sg["outb"])
            w_all.append(sg["W"].reshape(-1))
            io += K; oo += U; wo += n_species * K * U
        self.seg = torch.tensor(rows, dtype=torch.int64, device=device)
        self.inb = torch.tensor(inb_all, dtype=torch.int64, device=device)
        self.outb = torch.tensor(outb_all, dtype=torch.int64, device=device)
        self.W = torch.cat(w_all).to(device=device, dtype=torch.float64).contiguous()
        self.nseg = len(segs)
        self.Dmax = max(sg["D"] for sg in segs)
        self.Umax = max(len(sg["outb"]) for sg in segs)
        self.in_dim, self.out_dim, self.out_tail = in_dim, out_dim, tuple(out_shape_tail)
        self.S = n_species
        self.zero_fill = zero_fill  # output not fully covered by segments -> must be zero-initialised

    def run(self, x, attrs):
        N = x.shape[0]
        x = x.reshape(N, self.in_dim)
        if x.stride(1) != 1:
            x = x.contiguous()
        alloc = torch.zeros if self.zero_fill else torch.empty
        y = alloc((N,) + self.out_tail, dtype=torch.float64, device=x.device)
        if N == 0:
            return y
        has_a = attrs is not None
        if has_a and attrs.stride(1) != 1:
            attrs = attrs.contiguous()
        br = _block_r(N * self.Dmax)
        grid = (triton.cdiv(N * self.Dmax, br), self.nseg, triton.cdiv(self.Umax, BLOCK_U))
        _seg_linear_kernel[grid](x, x.stride(0), y, self.out_dim, attrs if has_a else x, attrs.stride(0) if has_a else 0,
                                 self.S if has_a else 1, self.W, self.seg, self.inb, self.outb, N,
                                 HAS_A=has_a, BR=br, BU=BLOCK_U, BK=BLOCK_K, num_warps=NUM_WARPS)
        return y


class _FusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, attrs, op):
        ctx.op = op
        if attrs is not None:
            ctx.save_for_backward(attrs)
        else:
            ctx.save_for_backward()
        return op.fwd.run(x, attrs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        saved = ctx.saved_tensors
        attrs = saved[0] if saved else None
        gx = ctx.op.bwd.run(g.contiguous(), attrs)
        return gx, None, None


class FusedNodeOp:
    """y = sum_s attrs[:,s] * L_s(x) for a map L given as segments; fwd and bwd plans."""

    def __init__(self, name, segs, in_dim, out_dim, out_tail, n_species, device, params, species):
        self.name = name
        self.species = species
        self.fwd = SegPlan(segs, in_dim, out_dim, out_tail, n_species, device, zero_fill=False)
        covered = sorted(i + m for sg in segs for i in sg["inb"] for m in range(sg["D"]))
        full_in = covered == list(range(in_dim))
        bsegs = [dict(D=sg["D"], inb=sg["outb"], outb=sg["inb"], W=sg["W"].transpose(1, 2).contiguous()) for sg in segs]
        self.bwd = SegPlan(bsegs, out_dim, in_dim, (in_dim,), n_species, device, zero_fill=not full_in)
        self.params = params  # (param, _version at install) -> stale-weight guard
        self.versions = [p._version for p in params]

    def __call__(self, x, attrs):
        if any(p._version != v for p, v in zip(self.params, self.versions)):
            raise FuseNodeError(f"{self.name}: source weights modified after install(); uninstall + reinstall")
        if attrs is not None and attrs.requires_grad:
            raise FuseNodeError(f"{self.name}: node_attrs requires grad; fused op gives no d/d(node_attrs)")
        if x.dtype != torch.float64:
            raise FuseNodeError(f"{self.name}: fp64 only, got {x.dtype}")
        return _FusedFn.apply(x, attrs if self.species else None, self)


# ------------------------------------------------------------------------------------------------------- weight folding
def _irrep_slices(irreps):
    out, off = [], 0
    for mul, ir in o3.Irreps(irreps):
        out.append((ir, mul, off))
        off += mul * ir.dim
    return out, off


@torch.no_grad()
def _extract(fn, irreps_in, irreps_out, n_species, layout, device):
    """fn(X [B,in_dim], s) -> [B,out_dim] (original modules). Returns segments + structural error."""
    ins, in_dim = _irrep_slices(irreps_in)
    outs, out_dim = _irrep_slices(irreps_out)
    if len({ir for ir, _, _ in outs}) != len(outs):
        raise FuseNodeError(f"output irreps not simplified: {irreps_out}")
    eye = torch.eye(in_dim, dtype=torch.float64, device=device)
    dense = [fn(eye, s) for s in range(n_species)]  # dense[s][i, j]: input basis i -> output j
    if layout == "reshape":
        muls = {mul for _, mul, _ in outs}
        if len(muls) != 1:
            raise FuseNodeError("reshape layout needs uniform mul")
        mulo = muls.pop()
        tot_d = sum(ir.dim for ir, _, _ in outs)
        out_tail = (mulo, tot_d)
    else:
        out_tail = (out_dim,)
    segs, doff = [], 0
    recon = [torch.zeros(in_dim, out_dim, dtype=torch.float64, device=device) for _ in range(n_species)]
    for ir, mulo, oo in outs:
        D = ir.dim
        inb = [io + v * D for (iri, muli, io) in ins if iri == ir for v in range(muli)]
        U = list(range(mulo))
        outb_flat = [oo + u * D for u in U]
        outb = [u * out_tail[1] + doff for u in U] if layout == "reshape" else outb_flat
        doff += D
        if not inb:
            raise FuseNodeError(f"no input for output irrep {ir}")
        ib = torch.tensor(inb, device=device)
        ob = torch.tensor(outb_flat, device=device)
        W = torch.stack([dense[s][ib][:, ob] for s in range(n_species)])  # m=0 component [S,K,U]
        for s in range(n_species):
            for mm in range(D):
                recon[s][(ib + mm)[:, None], (ob + mm)[None, :]] = W[s]
        segs.append(dict(D=D, inb=inb, outb=outb, W=W.cpu()))
    err = max(float((recon[s] - dense[s]).abs().max()) for s in range(n_species))
    scale = max(float(dense[s].abs().max()) for s in range(n_species))
    rel = err / max(scale, 1e-300)
    if rel > STRUCT_TOL:
        raise FuseNodeError(f"fold structure check failed: rel err {rel:.3e} (map is not per-l/m-diagonal as assumed)")
    return segs, in_dim, out_dim, out_tail, rel


# ------------------------------------------------------------------------------------------------- forward overrides
def _forward_L0(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index, cutoff=None, lammps_natoms=(0, 0),
                lammps_class=None, first_layer=False):
    # copy of mace 0.3.17 RealAgnosticInteractionBlock.forward; last three ops replaced by the fused kernel
    n_real = lammps_natoms[0] if lammps_class is not None else None
    node_feats = self.linear_up(node_feats)
    node_feats = self.handle_lammps(node_feats, lammps_class=lammps_class, lammps_natoms=lammps_natoms,
                                    first_layer=first_layer)
    tp_weights = self.conv_tp_weights(edge_feats)
    if cutoff is not None:
        tp_weights = tp_weights * cutoff
    if hasattr(self, "conv_fusion"):
        message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
    else:
        from mace.tools.scatter import scatter_sum
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0])
    message = self.truncate_ghosts(message, n_real)
    node_attrs = self.truncate_ghosts(node_attrs, n_real)
    ops = self.__dict__[_OPS]
    return ops["out"](message, node_attrs), None  # == reshape(skip_tp(linear(message)/avg, node_attrs))


def _forward_L1(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index, cutoff=None, lammps_class=None,
                lammps_natoms=(0, 0), first_layer=False):
    # copy of mace 0.3.17 RealAgnosticResidualInteractionBlock.forward; skip_tp and linear//avg+reshape fused
    n_real = lammps_natoms[0] if lammps_class is not None else None
    ops = self.__dict__[_OPS]
    sc = ops["sc"](node_feats, node_attrs)  # == skip_tp(node_feats, node_attrs)
    node_feats = self.linear_up(node_feats)
    node_feats = self.handle_lammps(node_feats, lammps_class=lammps_class, lammps_natoms=lammps_natoms,
                                    first_layer=first_layer)
    tp_weights = self.conv_tp_weights(edge_feats)
    if cutoff is not None:
        tp_weights = tp_weights * cutoff
    if hasattr(self, "conv_fusion"):
        message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
    else:
        from mace.tools.scatter import scatter_sum
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0])
    message = self.truncate_ghosts(message, n_real)
    node_attrs = self.truncate_ghosts(node_attrs, n_real)
    sc = self.truncate_ghosts(sc, n_real)
    return ops["out"](message, None), sc  # == reshape(linear(message)/avg)


# sha256 of inspect.getsource(cls.forward) for mace 0.3.17 (recorded at development; any drift -> refuse to install)
_SRC_SHA = {
    "RealAgnosticInteractionBlock": "0a9b26328a7533f380b45af53e78f60e98ffed022cb9b7c136f781ddefefdf97",
    "RealAgnosticResidualInteractionBlock": "5148c60b11fd7dcb8fca624aa0447b9e508fcb4d7869a2b3d79e1ac14f768b62",
}


def _src_sha(cls):
    return hashlib.sha256(inspect.getsource(cls.forward).encode()).hexdigest()


def _check_block(i, block):
    from e3nn.o3 import Linear as E3Linear, FullyConnectedTensorProduct as E3FCTP
    from mace.modules import blocks as MB
    cls = type(block)
    name = cls.__name__
    if cls not in (MB.RealAgnosticInteractionBlock, MB.RealAgnosticResidualInteractionBlock):
        raise FuseNodeError(f"interactions.{i}: unsupported class {cls.__module__}.{name}")
    want = _SRC_SHA[name]
    if want is not None and _src_sha(cls) != want:
        raise FuseNodeError(f"interactions.{i}: {name}.forward source differs from mace 0.3.17 copy; refusing")
    if "forward" in block.__dict__:
        raise FuseNodeError(f"interactions.{i}: forward already overridden on instance (already installed?)")
    for n in ("linear", "skip_tp"):
        mod = getattr(block, n)
        if type(mod) not in (E3Linear, E3FCTP):
            raise FuseNodeError(f"interactions.{i}.{n}: {type(mod)} is not plain e3nn (cueq/oeq linear unsupported)")
    if getattr(block.reshape, "cueq_config", None) is not None:
        raise FuseNodeError(f"interactions.{i}: reshape has cueq_config (ir_mul layout unsupported)")
    b_ = getattr(block.linear, "bias", None)
    if isinstance(b_, torch.Tensor) and b_.numel():
        raise FuseNodeError(f"interactions.{i}.linear has biases")
    if block.node_attrs_irreps != o3.Irreps(f"{block.node_attrs_irreps.dim}x0e"):
        raise FuseNodeError(f"interactions.{i}: node_attrs irreps {block.node_attrs_irreps} not scalar")


def _build_ops(block, device):
    from mace.modules import blocks as MB
    avg = block.avg_num_neighbors
    S = block.node_attrs_irreps.dim
    ops, info = {}, {}
    if type(block) is MB.RealAgnosticInteractionBlock:
        def f(X, s):
            a = torch.zeros(X.shape[0], S, dtype=X.dtype, device=X.device); a[:, s] = 1.0
            return block.skip_tp(block.linear(X) / avg, a)
        segs, di, do, tail, rel = _extract(f, block.linear.irreps_in, block.irreps_out, S, "reshape", device)
        params = list(block.linear.parameters()) + list(block.skip_tp.parameters())
        ops["out"] = FusedNodeOp(f"L0.linear/avg.skip_tp.reshape", segs, di, do, tail, S, device, params, True)
        info["out"] = dict(fold_struct_rel_err=rel, in_dim=di, out=tail, species=S)
    else:
        def f(X, s):
            return block.linear(X) / avg
        segs, di, do, tail, rel = _extract(f, block.linear.irreps_in, block.linear.irreps_out, 1, "reshape", device)
        ops["out"] = FusedNodeOp("L1.linear/avg.reshape", segs, di, do, tail, 1, device,
                                 list(block.linear.parameters()), False)
        info["out"] = dict(fold_struct_rel_err=rel, in_dim=di, out=tail, species=1)

        def g(X, s):
            a = torch.zeros(X.shape[0], S, dtype=X.dtype, device=X.device); a[:, s] = 1.0
            return block.skip_tp(X, a)
        segs, di, do, tail, rel = _extract(g, block.skip_tp.irreps_in1, block.skip_tp.irreps_out, S, "flat", device)
        ops["sc"] = FusedNodeOp("L1.skip_tp", segs, di, do, tail, S, device, list(block.skip_tp.parameters()), True)
        info["sc"] = dict(fold_struct_rel_err=rel, in_dim=di, out=tail, species=S)
    return ops, info


# ---------------------------------------------------------------------------------------------------------------- API
def install(model):
    from mace.modules import blocks as MB
    if getattr(model, _STATE, None):
        raise FuseNodeError("fuse7_node already installed")
    p0 = next(model.parameters())
    if p0.dtype != torch.float64 or p0.device.type != "cuda":
        raise FuseNodeError(f"need fp64 CUDA model, got {p0.dtype} {p0.device}")
    for i, b in enumerate(model.interactions):
        _check_block(i, b)
    handle = dict(blocks=[], info={})
    for i, b in enumerate(model.interactions):
        ops, info = _build_ops(b, p0.device)
        fwd = _forward_L0 if type(b) is MB.RealAgnosticInteractionBlock else _forward_L1
        b.__dict__[_OPS] = ops
        b.__dict__["forward"] = types.MethodType(fwd, b)
        handle["blocks"].append(i)
        handle["info"][f"interactions.{i}"] = info
    object.__setattr__(model, _STATE, handle)
    return handle


def uninstall(model, handle=None):
    h = getattr(model, _STATE, None)
    if not h:
        return False
    for i in h["blocks"]:
        b = model.interactions[i]
        b.__dict__.pop("forward", None)
        b.__dict__.pop(_OPS, None)
    object.__delattr__(model, _STATE)
    return True


def is_installed(model):
    return bool(getattr(model, _STATE, None))


def describe():
    return dict(
        module="scripts/fuse7_node/fused_node.py",
        mechanism="instance-level forward override (types.MethodType in block.__dict__) on interactions.{0,1}; "
                  "stock submodules (linear, skip_tp, reshape) are left in place but unused; uninstall pops the override",
        replaced={
            "interactions.0": "linear(message) / avg_num_neighbors -> skip_tp(., node_attrs) -> reshape  ==> 1 kernel "
                              "(per-species folded W[s,l] 128x128, l=0..3, output [N,128,16])",
            "interactions.1 (message)": "linear(message) / avg_num_neighbors -> reshape ==> 1 kernel (1/avg folded; "
                                        "per-l 256/384/384/256 x 128, output [N,128,16])",
            "interactions.1 (sc)": "skip_tp(node_feats, node_attrs) ==> 1 kernel (per-species 128x128 on 0e)",
        },
        kernel="_seg_linear_kernel (Triton, fp64 tl.dot), segment table over irreps, species tile-skip",
        launches_fwd=3, launches_bwd=4,  # bwd: 3 kernels + 1 zero-fill memset for L1 sc (1o input part has zero grad)
        backward="custom autograd.Function, grad wrt x only (W^T gathered by species); once_differentiable",
        constraints="fp64, CUDA, e3nn (non-cueq) linear/skip_tp, mul_ir reshape; weights snapshotted at install "
                    "(guarded by param._version)",
    )
