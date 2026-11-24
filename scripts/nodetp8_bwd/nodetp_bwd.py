"""Node-centric (source-centric) fused fp64 backward of MACE's uvu conv (gather + TP + scatter_sum).

bwd(layer_cfg, grad_out, x, y, w, edge_index, csr) -> (gx [N,d1], gy [E,d2], gw [E,W])

Design (one fused kernel per layer, CUDA via load_inline, code generated per LayerCfg with the fp64 CG values as exact
literals (repr round-trips), path weights folded at run time as in e3nn: pw * sum(...)):
  * grid = N CTAs (one per SOURCE node n), blockDim = mul (=128): thread u owns channel u.
  * x[n, u, :] and the gx[n, u, :] accumulators live in registers for the whole CTA (read/written once per node);
  * the CTA walks its outgoing edges e = perm_src[rowptr_src[n] : rowptr_src[n+1]]; per edge it reads y[e] (broadcast),
    w[e, p*mul+u] (coalesced) and grad_out[dst(e), :] (the dominant traffic; same as any gather-based backward), and
    for every CG nonzero (i,j,k,v) of path p, with h_k = pw*g[dst,u,k], s_k = w_pu*h_k:
        gw_pu += v x_i y_j h_k ;  gy_j += v x_i s_k (partial over u) ;  gx_i += v y_j s_k
  * gw[e] written directly (each edge belongs to exactly one source -> no atomics); gy[e] = sum over u: warp shuffle
    reduction + 1 smem round per edge (double buffered, one __syncthreads per edge); gx written at the end. No atomics.
  Why fused (not edge kernel + src kernel): gx, gy, gw all need the same per-(e,u,path) products of g[dst] with x[src];
  a split design would read grad_out[dst] (the dominant E*d_out*8 B stream) twice; src-centric order makes x/gx
  register-resident. A dst-centric split would need atomics for gx.
No host syncs, launches on the current stream, shapes depend only on (N, E); E=0 -> gx zeros, gy/gw [0,*].
"""
import os
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

_MODS = {}


def _gen_kernel(cfg, name):
    m = cfg.mul
    assert m % 32 == 0 and m <= 1024
    nw = m // 32
    L = []
    a = L.append
    a(f'extern "C" __global__ void __launch_bounds__({m}) {name}(const double* __restrict__ x, const double* __restrict__ y,')
    a('    const double* __restrict__ w, const double* __restrict__ g, const long long* __restrict__ ei,')
    a('    const long long* __restrict__ perm, const int* __restrict__ rowptr, double* __restrict__ gx,')
    a('    double* __restrict__ gy, double* __restrict__ gw, long long E) {')
    a(f'  __shared__ double red[2][{nw}][{cfg.d2}];')
    a('  const int n = blockIdx.x; const int u = threadIdx.x; const int lane = u & 31; const int wid = u >> 5;')
    # x registers / gx accumulators
    for ii, (mul, l, p, off, dim) in enumerate(cfg.x_irreps):
        for i in range(dim):
            a(f'  const double x{ii}_{i} = __ldg(x + (long long)n * {cfg.d1} + {off} + u * {dim} + {i}); double gx{ii}_{i} = 0.0;')
    a('  const int t0 = rowptr[n], t1 = rowptr[n + 1];')
    a('  int buf = 0;')
    a('  for (int t = t0; t < t1; ++t) {')
    a('    const long long e = perm[t]; const long long d = ei[E + e];')
    for j in range(cfg.d2):
        a(f'    const double y{j} = __ldg(y + e * {cfg.d2} + {j}); double gy{j} = 0.0;')
    used_out = sorted({P["io"] for P in cfg.paths})
    for io in used_out:
        mul, l, p, off, dim = cfg.out_irreps[io]
        for k in range(dim):
            a(f'    const double g{io}_{k} = __ldg(g + d * {cfg.d_out} + {off} + u * {dim} + {k});')
    for pi, P in enumerate(cfg.paths):
        a(f'    {{ // path {pi}: l=({P["l1"]},{P["l2"]},{P["l3"]}) pw={P["path_weight"]!r}')
        a(f'      const double pw = {P["path_weight"]!r}; const double wp = __ldg(w + e * {cfg.W} + {P["w_off"]} + u);')
        dim3 = 2 * P["l3"] + 1
        for k in range(dim3):
            a(f'      const double h{k} = pw * g{P["io"]}_{k}; const double s{k} = wp * h{k};')
        a('      double aw = 0.0;')
        yo = P["y_off"]
        for (i, j, k, v) in P["nz"]:
            vs = repr(v)
            a(f'      aw = fma({vs} * x{P["i1"]}_{i}, y{yo + j} * h{k}, aw);'
              f' gy{yo + j} = fma({vs} * x{P["i1"]}_{i}, s{k}, gy{yo + j});'
              f' gx{P["i1"]}_{i} = fma({vs} * y{yo + j}, s{k}, gx{P["i1"]}_{i});')
        a(f'      gw[e * {cfg.W} + {P["w_off"]} + u] = aw;')
        a('    }')
    # gy reduction over u
    for j in range(cfg.d2):
        a(f'    for (int o = 16; o > 0; o >>= 1) gy{j} += __shfl_xor_sync(0xffffffffu, gy{j}, o);')
    a('    if (lane == 0) {')
    for j in range(cfg.d2):
        a(f'      red[buf][wid][{j}] = gy{j};')
    a('    }')
    a('    __syncthreads();')
    a(f'    if (u < {cfg.d2}) {{ double s = 0.0; for (int q = 0; q < {nw}; ++q) s += red[buf][q][u]; gy[e * {cfg.d2} + u] = s; }}')
    a('    buf ^= 1;')
    a('  }')
    for ii, (mul, l, p, off, dim) in enumerate(cfg.x_irreps):
        for i in range(dim):
            a(f'  gx[(long long)n * {cfg.d1} + {off} + u * {dim} + {i}] = gx{ii}_{i};')
    a('}')
    return "\n".join(L)


_CPP = "void launch(at::Tensor x, at::Tensor y, at::Tensor w, at::Tensor g, at::Tensor ei, at::Tensor perm, at::Tensor rowptr, at::Tensor gx, at::Tensor gy, at::Tensor gw);"


def _get_mod(cfg):
    h = hashlib.sha1(cfg.key.encode()).hexdigest()[:12]
    if h in _MODS:
        return _MODS[h]
    name = f"nodetp_bwd_{h}"
    src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n"
    src += _gen_kernel(cfg, "kern") + "\n"
    src += f"""
void launch(at::Tensor x, at::Tensor y, at::Tensor w, at::Tensor g, at::Tensor ei, at::Tensor perm, at::Tensor rowptr,
            at::Tensor gx, at::Tensor gy, at::Tensor gw) {{
  const int N = (int)x.size(0); const long long E = y.size(0);
  if (N == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<N, {cfg.mul}, 0, stream>>>(x.data_ptr<double>(), y.data_ptr<double>(), w.data_ptr<double>(), g.data_ptr<double>(),
      (const long long*)ei.data_ptr<int64_t>(), (const long long*)perm.data_ptr<int64_t>(), rowptr.data_ptr<int>(),
      gx.data_ptr<double>(), gy.data_ptr<double>(), gw.data_ptr<double>(), E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""
    mod = load_inline(name=name, cpp_sources=_CPP, cuda_sources=src, functions=["launch"],
                      extra_cuda_cflags=["-O3", "-fmad=true", "-arch=sm_90"], verbose=False)
    _MODS[h] = mod
    return mod


def bwd(cfg, grad_out, x, y, w, edge_index, csr):
    N, E = x.shape[0], y.shape[0]
    assert x.dtype == torch.float64 and x.is_cuda
    assert grad_out.shape == (N, cfg.d_out) and x.shape[1] == cfg.d1 and y.shape[1] == cfg.d2 and w.shape == (E, cfg.W)
    gx = torch.empty((N, cfg.d1), dtype=x.dtype, device=x.device)
    gy = torch.empty((E, cfg.d2), dtype=x.dtype, device=x.device)
    gw = torch.empty((E, cfg.W), dtype=x.dtype, device=x.device)
    if N == 0:
        return gx, gy, gw
    ei = edge_index if edge_index.dtype == torch.int64 else edge_index.long()
    _get_mod(cfg).launch(x.contiguous(), y.contiguous(), w.contiguous(), grad_out.contiguous(), ei.contiguous(),
                         csr["perm_src"].long(), csr["rowptr_src"].int().contiguous(), gx, gy, gw)
    return gx, gy, gw
