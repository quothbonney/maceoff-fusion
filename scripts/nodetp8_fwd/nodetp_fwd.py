"""Node-centric (destination-owned) FP64 uvu tensor-product convolution forward for MACE-OFF23 (agent B, phase 8).

    out[n] = sum_{e: dst(e)=n} TP_uvu(x[src(e)], y[e], w[e])     (MACE conv_tp + gather + scatter_sum semantics)

API (INTERFACES.md):
    cfg = make_cfg(conv_tp)                       # NodeCfg (B's equivalent of C's layer_cfg); also accepts C's dicts
    csr = build_csr(edge_index, N)                # local stub, same signature/fields as C's scripts/nodetp8_bwd/csr.py
    out = fwd(cfg, x, y, w, edge_index, csr)      # [N, d_out] fp64, current stream, no host sync, graph-capturable

Kernel: one CTA of 128 threads per destination node (thread u = channel u; uvu is channel-diagonal). Each CTA walks its
dst-sorted incoming edges (perm_dst/rowptr_dst), loads x[src(e), :, u] (all in1 irreps of channel u), y[e] (16, broadcast
loads), w[e, k*128+u] (coalesced), and accumulates out[n, :, u] (d_out/128 values) in registers; each output written once,
no atomics, zero in-degree -> zeros. Next-edge operands are prefetched into registers (software pipelining).
CG: o3.wigner_3j(l1,l2,l3, float64) * instruction.path_weight, nonzeros baked into the generated source as %.17g literals
(exact fp64 round-trip). Products x_i*y_j are shared between paths with the same (i_in1, i_in2).
"""
import os
import hashlib
import torch
from e3nn import o3

MUL = 128
_EXT = {}


class NodeCfg:
    def __init__(self, conv_tp):
        tp = conv_tp
        self.irreps_in1, self.irreps_in2, self.irreps_out = str(tp.irreps_in1), str(tp.irreps_in2), str(tp.irreps_out)
        self.d1, self.d2, self.d_out, self.W = tp.irreps_in1.dim, tp.irreps_in2.dim, tp.irreps_out.dim, tp.weight_numel
        for irs in (tp.irreps_in1, tp.irreps_out):
            assert all(mul == MUL for mul, _ in irs), irs
        assert all(mul == 1 for mul, _ in tp.irreps_in2), tp.irreps_in2
        assert self.W == MUL * len(tp.instructions)
        off1 = [s.start for s in tp.irreps_in1.slices()]
        off2 = [s.start for s in tp.irreps_in2.slices()]
        off3 = [s.start for s in tp.irreps_out.slices()]
        self.paths = []
        for k, ins in enumerate(tp.instructions):
            assert ins.connection_mode == "uvu" and ins.has_weight and tuple(ins.path_shape) == (MUL, 1)
            ir1, ir2, ir3 = tp.irreps_in1[ins.i_in1].ir, tp.irreps_in2[ins.i_in2].ir, tp.irreps_out[ins.i_out].ir
            C = o3.wigner_3j(ir1.l, ir2.l, ir3.l, dtype=torch.float64) * float(ins.path_weight)
            nz = [(i, j, kk, float(C[i, j, kk])) for i in range(ir1.dim) for j in range(ir2.dim)
                  for kk in range(ir3.dim) if C[i, j, kk] != 0.0]
            self.paths.append(dict(k=k, i_in1=ins.i_in1, i_in2=ins.i_in2, i_out=ins.i_out, l=(ir1.l, ir2.l, ir3.l),
                                   dims=(ir1.dim, ir2.dim, ir3.dim), off1=off1[ins.i_in1], off2=off2[ins.i_in2],
                                   off3=off3[ins.i_out], woff=k * MUL, path_weight=float(ins.path_weight), cg=nz))
        self.nnz = sum(len(p["cg"]) for p in self.paths)

    def key(self):
        return hashlib.sha1(repr((self.irreps_in1, self.irreps_in2, self.irreps_out,
                                  [(p["off1"], p["off2"], p["off3"], p["woff"], p["dims"], p["cg"])
                                   for p in self.paths])).encode()).hexdigest()[:12]

    def flops_per_edge(self):
        """fp64 flops per edge (all 128 channels) of the generated code: x*y products + CG FMAs + w FMAs."""
        pairs = {}
        for p in self.paths:
            pairs.setdefault((p["i_in1"], p["i_in2"]), set()).update((i, j) for i, j, _, _ in p["cg"])
        prod = sum(len(s) for s in pairs.values())
        fma = sum(len(p["cg"]) for p in self.paths) + sum(p["dims"][2] for p in self.paths)
        return MUL * (prod + 2 * fma)


def make_cfg(conv_tp):
    return NodeCfg(conv_tp)


def _as_cfg(layer_cfg):
    if isinstance(layer_cfg, NodeCfg):
        return layer_cfg
    if isinstance(layer_cfg, torch.nn.Module):
        c = getattr(layer_cfg, "_nodetp8_cfg", None)
        if c is None:
            c = NodeCfg(layer_cfg)
            object.__setattr__(layer_cfg, "_nodetp8_cfg", c)
        return c
    if isinstance(layer_cfg, dict):  # C's layer_cfg: carries the conv_tp module or a NodeCfg
        for k in ("node_cfg", "b_cfg", "conv_tp", "module"):
            if k in layer_cfg:
                return _as_cfg(layer_cfg[k])
    for k in ("node_cfg", "b_cfg", "conv_tp", "module"):
        if hasattr(layer_cfg, k):
            return _as_cfg(getattr(layer_cfg, k))
    raise TypeError(f"cannot derive NodeCfg from {type(layer_cfg)}")


# ------------------------------------------------------------------------------------------------------------ CSR
def build_csr(edge_index, N):
    """Local stub matching INTERFACES (C owns the canonical scripts/nodetp8_bwd/csr.py). GPU-only ops on fixed shapes:
    stable sort + searchsorted (no bincount: its output size is data dependent -> host sync)."""
    dev = edge_index.device
    out = {}
    nodes = torch.arange(N + 1, device=dev, dtype=edge_index.dtype)
    for name, row in (("dst", edge_index[1]), ("src", edge_index[0])):
        srt, perm = torch.sort(row, stable=True)
        out[f"perm_{name}"] = perm.to(torch.int32)
        out[f"rowptr_{name}"] = torch.searchsorted(srt, nodes).to(torch.int32)
    return out


# ------------------------------------------------------------------------------------------------------- codegen
def _lit(v):
    return "%.17g" % v


def partition(paths, PG):
    """path groups (LunaTP-style TP-path grouping): LPT on accumulator count (registers), tie-break CG nnz."""
    groups = [[] for _ in range(PG)]
    load = [0.0] * PG
    for p in sorted(paths, key=lambda p: (-p["dims"][2], -len(p["cg"]))):
        gi = min(range(PG), key=lambda i: load[i])
        groups[gi].append(p)
        load[gi] += p["dims"][2] + 1e-3 * len(p["cg"])
    return [sorted(gp, key=lambda p: p["k"]) for gp in groups if gp]


def gen_group(cfg, P, fname):
    """device function for one path group: edge loop (slice g of G), register accumulators, smem reduction, store."""
    NACC = sum(p["dims"][2] for p in P)
    abase, b = {}, 0
    for p in P:
        abase[p["k"]] = b
        b += p["dims"][2]
    in1 = sorted({(p["i_in1"], p["off1"], p["dims"][0]) for p in P})
    xreg, nx = {}, 0
    for i1, off, d in in1:
        xreg[i1] = (nx, off, d)
        nx += d
    in2 = sorted({(p["i_in2"], p["off2"], p["dims"][1]) for p in P})
    yreg, ny = {}, 0
    for i2, off, d in in2:
        yreg[i2] = (ny, off, d)
        ny += d
    L = []
    a = L.append
    a("template <typename IP, typename IE, int G>")
    a(f"__device__ __forceinline__ void {fname}(const double* __restrict__ x, const double* __restrict__ y,")
    a("    const double* __restrict__ w, const IE* __restrict__ src, const IP* __restrict__ perm,")
    a("    double* __restrict__ out, const int n, const int u, const int g, const int beg, const int end,")
    a("    double* red, const int barid) {")
    a(f"  double acc[{NACC}];")
    a("  #pragma unroll")
    a(f"  for (int q = 0; q < {NACC}; ++q) acc[q] = 0.0;")

    def load(sfx, pidx):
        t = [f"{{ const long long e = (long long)perm[{pidx}]; const long long sn = (long long)src[e];",
             f"  const double* xr = x + sn * {cfg.d1}; const double* wr = w + e * {cfg.W}; const double* yr = y + e * {cfg.d2};"]
        for i1, (b0, off, d) in xreg.items():
            for i in range(d):
                t.append(f"  xv{sfx}[{b0 + i}] = __ldg(xr + {off} + u * {d} + {i});")
        for j, p in enumerate(P):
            t.append(f"  wv{sfx}[{j}] = __ldg(wr + {p['woff']} + u);")
        for i2, (b0, off, d) in yreg.items():
            for j in range(d):
                t.append(f"  yv{sfx}[{b0 + j}] = __ldg(yr + {off + j});")
        t.append("}")
        return "\n".join(t)
    for sfx in ("", "n"):
        a(f"  double xv{sfx}[{nx}], wv{sfx}[{len(P)}], yv{sfx}[{ny}];")
    a("  if (beg + g < end) {")
    a(load("n", "beg + g"))
    a("  }")
    a("  for (int p = beg + g; p < end; p += G) {")
    a("    #pragma unroll")
    a(f"    for (int q = 0; q < {nx}; ++q) xv[q] = xvn[q];")
    a("    #pragma unroll")
    a(f"    for (int q = 0; q < {len(P)}; ++q) wv[q] = wvn[q];")
    a("    #pragma unroll")
    a(f"    for (int q = 0; q < {ny}; ++q) yv[q] = yvn[q];")
    a(f"    if (p + G < end) {load('n', 'p + G')}")
    pairs = {}
    for p in P:
        pairs.setdefault((p["i_in1"], p["i_in2"]), set()).update((i, j) for i, j, _, _ in p["cg"])
    for (i1, i2), ij in pairs.items():
        for i, j in sorted(ij):
            a(f"    const double p{i1}_{i2}_{i}_{j} = xv[{xreg[i1][0] + i}] * yv[{yreg[i2][0] + j}];")
    for jp, p in enumerate(P):
        for kk in range(p["dims"][2]):
            terms = [(i, j, c) for i, j, k3, c in p["cg"] if k3 == kk]
            if not terms:
                continue
            i, j, c = terms[0]
            expr = f"{_lit(c)} * p{p['i_in1']}_{p['i_in2']}_{i}_{j}"
            for i, j, c in terms[1:]:
                expr = f"fma({_lit(c)}, p{p['i_in1']}_{p['i_in2']}_{i}_{j}, {expr})"
            a(f"    acc[{abase[p['k']] + kk}] = fma(wv[{jp}], {expr}, acc[{abase[p['k']] + kk}]);")
    a("  }")
    a("  if (G > 1) {")
    a("    for (int gg = G - 1; gg >= 1; --gg) {")
    a("      if (g == gg) {")
    a("        #pragma unroll")
    a(f"        for (int q = 0; q < {NACC}; ++q) red[q * 128 + u] = acc[q];")
    a("      }")
    a("      asm volatile(\"bar.sync %0, %1;\" :: \"r\"(barid), \"r\"(128 * G) : \"memory\");")
    a("      if (g == 0) {")
    a("        #pragma unroll")
    a(f"        for (int q = 0; q < {NACC}; ++q) acc[q] += red[q * 128 + u];")
    a("      }")
    a("      asm volatile(\"bar.sync %0, %1;\" :: \"r\"(barid), \"r\"(128 * G) : \"memory\");")
    a("    }")
    a("    if (g != 0) return;")
    a("  }")
    a(f"  double* o = out + (long long)n * {cfg.d_out};")
    for p in P:
        d3 = p["dims"][2]
        for kk in range(d3):
            a(f"  o[{p['off3']} + u * {d3} + {kk}] = acc[{abase[p['k']] + kk}];")
    a("}")
    return "\n".join(L), NACC


def gen_kernel(cfg, name, PG):
    groups = partition(cfg.paths, PG)
    src, offs, off = [], [], 0
    for gi, P in enumerate(groups):
        code, nacc = gen_group(cfg, P, f"{name}_g{gi}")
        src.append(code)
        offs.append(off)
        off += nacc * MUL
    a = src.append
    a("template <typename IP, typename IE, int G>")
    a(f"__global__ void __launch_bounds__(128 * {len(groups)} * G) {name}(const double* __restrict__ x, const double* __restrict__ y,")
    a("    const double* __restrict__ w, const IE* __restrict__ src, const IP* __restrict__ perm,")
    a("    const int* __restrict__ rowptr, double* __restrict__ out) {")
    a("  extern __shared__ double red[];")
    a(f"  const int n = blockIdx.x; const int u = threadIdx.x & 127; const int pg = (threadIdx.x >> 7) % {len(groups)};")
    a(f"  const int g = threadIdx.x / (128 * {len(groups)});")
    a("  const int beg = rowptr[n], end = rowptr[n + 1];")
    for gi in range(len(groups)):
        a(f"  {'if' if gi == 0 else 'else if'} (pg == {gi}) {name}_g{gi}<IP, IE, G>(x, y, w, src, perm, out, n, u, g, beg, end, red + {offs[gi]}, {gi + 1});")
    a("}")
    return "\n".join(src), len(groups), groups


def _launcher(name, nacc, NG):
    gs = [g for g in (1, 2, 4, 8) if g * NG <= 8]
    gsel = " else ".join(f"if (G <= {g}) L2(IP, IE, {g})" for g in gs[:-1]) + f" else L2(IP, IE, {gs[-1]})"
    return f"""
void launch_{name}(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor ei, torch::Tensor perm,
                   torch::Tensor rowptr, torch::Tensor out, int64_t G) {{
  const int N = out.size(0);
  if (N == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  const double* xp = x.data_ptr<double>(); const double* yp = y.data_ptr<double>(); const double* wp = w.data_ptr<double>();
  const int* rp = rowptr.data_ptr<int>(); double* op = out.data_ptr<double>();
  const size_t sh = (size_t){nacc} * 128 * sizeof(double);
  #define L2(IP, IE, GG) {{ if (GG > 1 && sh > 48 * 1024) cudaFuncSetAttribute({name}<IP, IE, GG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh); \\
     {name}<IP, IE, GG><<<N, 128 * {NG} * GG, GG > 1 ? sh : 0, st>>>(xp, yp, wp, ei.data_ptr<IE>(), perm.data_ptr<IP>(), rp, op); }}
  #define L(IP, IE) {{ {gsel} }}
  if (perm.scalar_type() == at::kInt) {{ if (ei.scalar_type() == at::kLong) L(int, int64_t) else L(int, int) }}
  else {{ if (ei.scalar_type() == at::kLong) L(int64_t, int64_t) else L(int64_t, int) }}
  #undef L
  #undef L2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""


def default_PG(cfg):
    if os.environ.get("NODETP_PG"):
        return int(os.environ["NODETP_PG"])
    return 1  # path grouping (PG=2,4) measured slower on H100 for both layers (sweep_v3.json)


def get_ext(cfg, PG=None):
    PG = default_PG(cfg) if PG is None else PG
    key = cfg.key() + f"p{PG}"
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load_inline
    name = f"nodetp_fwd_{key}"
    cuda_src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n" \
        + gen_kernel(cfg, "k_" + key, PG)[0] + _launcher("k_" + key, cfg.d_out // MUL, len(partition(cfg.paths, PG)))
    cpp_src = f"void launch_k_{key}(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
    ext = load_inline(name=name, cpp_sources=cpp_src, cuda_sources=cuda_src, functions=[f"launch_k_{key}"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90",
                                         "-Xptxas=-v"] if os.environ.get("NODETP_PTXAS_V") else
                      ["-O3", "-gencode=arch=compute_90,code=sm_90"],
                      verbose=bool(os.environ.get("NODETP_PTXAS_V")))
    _EXT[key] = getattr(ext, f"launch_k_{key}")
    return _EXT[key]


def choose_G(N, E, cfg=None):
    """edge slices per destination CTA, from host-known shapes only (no sync). Tuned on H100 (results/nodetp8_fwd/
    sweep_v3.json): L0 (16 acc regs/thread): 4 for tiny graphs, else 2; L1 (40 acc, 238 regs): 2 if N < 400 else 1."""
    if os.environ.get("NODETP_G"):
        return int(os.environ["NODETP_G"])
    big = cfg is not None and cfg.d_out // MUL > 24
    if big:
        return 2 if N < 400 else 1
    return 4 if N < 64 else 2


def max_G(cfg, PG=None):
    return max(1, 8 // len(partition(cfg.paths, default_PG(cfg) if PG is None else PG)))


def fwd(layer_cfg, x, y, w, edge_index, csr, out=None, G=None, PG=None):
    cfg = _as_cfg(layer_cfg)
    N = x.shape[0]
    assert x.dtype == y.dtype == w.dtype == torch.float64
    assert x.shape[1] == cfg.d1 and y.shape[1] == cfg.d2 and w.shape[1] == cfg.W
    if out is None:
        out = torch.empty((N, cfg.d_out), dtype=torch.float64, device=x.device)
    rp = csr["rowptr_dst"]
    if rp.dtype != torch.int32:
        rp = rp.to(torch.int32)
    get_ext(cfg, PG)(x.contiguous(), y.contiguous(), w.contiguous(), edge_index.contiguous(), csr["perm_dst"].contiguous(),
                 rp.contiguous(), out, choose_G(N, edge_index.shape[1], cfg) if G is None else G)
    return out
