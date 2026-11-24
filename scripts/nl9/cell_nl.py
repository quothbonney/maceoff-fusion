"""Phase 9 / nl9: GPU cell-list neighbour builder emitting dst-sorted edges + CSR directly (no sort, no host sync).

API (results/phase9/INTERFACES.md):
    NL = CellNeighborList(r_max, pbc, cell, n_atoms, E_cap, device, bin_cap=None, bins_per_dim=None)
    out = NL.build(positions)   # positions [N,3] fp64 cuda contiguous (caller's unwrapped coordinates)
  out (persistent buffers, OVERWRITTEN by the next build; fixed shapes for fixed (N, E_cap)):
    edge_index [2,E_cap] int64   row0 = sender i, row1 = receiver j (MACE/matscipy: D = pos[j]-pos[i]+S@cell),
                                 sorted by receiver (dst), within a row by sender (unique per row), padding at the end
    unit_shifts [E_cap,3] fp64, shifts [E_cap,3] fp64 (= unit_shifts @ cell with cuBLAS's FMA order, see below)
    rowptr_dst [N+1] int32       rowptr_dst[N] = E (real edges; padding excluded), clamped to E_cap on overflow
    rowptr_src [N+1] int32       == rowptr_dst (list is symmetric)
    perm_dst [E_cap] int64       identity (edges already dst-sorted)
    perm_src [E_cap] int64       src-sorted order == build_csr's stable sort of edge_index[0] restricted to [0,E);
                                 identity on padding [E, E_cap)
    rowptr_dst_pad / rowptr_src_pad [N+1] int32: same with last entry E_cap (padding attached to node N-1) ->
                                 bitwise equal to nodetp8_bwd.csr.build_csr(edge_index, N) on the padded edge_index
    n_edges 0-d int64 (true E, may exceed E_cap), overflow 0-d int32 bit flags (1: E > E_cap, 2: bin overflow,
    4: reverse edge not found (internal error)), max_bin 0-d int32 (max atoms seen in one bin, for resizing)
    cell [3,3], rcell [3,3], volume [1] fp64: exactly as proto_graph / stock (non-periodic: per-step extended cell
    computed on device)
  Padding edges e in [E, E_cap): self-edge (N-1 -> N-1), unit_shift (m,0,0), shift = m*cell[0];
    periodic m = max(1, ceil(2 r_max/|cell[0]|)); non-periodic m = 1 (extended cell row 0 has length >= 2 r_max + 1).
    |D| >= r_max -> MACE's polynomial envelope is exactly 0 -> zero energy/force/stress contribution (also when MACE
    recomputes shifts = unit_shifts @ displaced cell for stress). Padding is the LAST node, so dst/src order holds.
  batch_dict(): BatchTemplate(calc, atoms).batch(out, positions) -> MACE batch dict (keys/dtypes/values of
    proto_graph.ProtoBuilder.build except edge order; padded to E_cap unless trim=True, which host-syncs).

Distance test = proto_graph's exactly: non-periodic d = pos[j]-pos[i]; periodic w = floor(pos @ inv(cell)) (wrap),
  pw = pos - w @ cell and shifts = S @ cell in-kernel as fma(a2,C2c, fma(a1,C1c, a0*C0c)) == cuBLAS fp64 [M,3]@[3,3]
  (probed bitwise for M = 1..300000, scripts/nl9/probe_mm2.py), i.e. == proto's torch matmuls,
  d = (pw[j]-pw[i]) + img@cell (27-entry table from the same torch matmul as proto), d2 = (d0^2 + d2^2) + d1^2
  (torch's reduction order for a last dim of 3, probed; __dmul_rn/__dadd_rn, no FMA), edge iff d2 < r_max*r_max (fp64); self-edge removed iff i==j and img==0;
  unit_shifts = (img - w[j]) + w[i].
Supported: pbc all False (bounding-box bins recomputed on device each build; bins >= r_max, at most bins_per_dim per
  axis -> bins grow for large/sparse extents; bin overflow flagged), pbc all True with every perpendicular cell height
  >= 3 bins of >= r_max (orthorhombic or triclinic). Otherwise NotImplementedError -> caller uses proto_graph.
"""
import math
import numpy as np
import torch

_CPP = r"""
#include <torch/extension.h>
void nl_build(torch::Tensor pos, torch::Tensor pw, torch::Tensor w, torch::Tensor geo, torch::Tensor ivtab,
              torch::Tensor bin_cnt, torch::Tensor bins, torch::Tensor atom_bin, torch::Tensor row_cnt,
              torch::Tensor rowptr, torch::Tensor rowptr_pad, torch::Tensor edge_index, torch::Tensor img_pack,
              torch::Tensor unit_shifts, torch::Tensor shifts, torch::Tensor perm_src, torch::Tensor n_edges,
              torch::Tensor flags, torch::Tensor cellbuf, torch::Tensor pw_mm, torch::Tensor wC,
              int64_t periodic, int64_t ortho, int64_t bin_cap, int64_t nb_cap, double rc2, double cutoff, double rbin,
              int64_t pad_m, int64_t stage);
"""

_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// geo layout (double): [0..8] inv(cell) row-major, [9..17] cell (search cell == returned cell when periodic),
//   [18..20] origin (non-periodic min), [21..23] bin width (non-periodic), [24..26] nb (as double),
//   [27..29] nb_cap per axis (non-periodic), [30] unused
// cellbuf (double): [0..8] cell, [9..17] rcell, [18] volume   (non-periodic: written each build)

__device__ __forceinline__ void atomic_or_i(int* p, int v) { atomicOr(p, v); }

// ---- stage 0: prep (single block): zero counters, non-periodic bbox / bins / extended cell ----
__global__ void k_prep(const double* __restrict__ pos, int N, double* geo, int* bin_cnt, int nbins_alloc,
                       int* flags, long long* n_edges, double* cellbuf, int periodic, double cutoff, double rbin) {
  __shared__ double smin[3][256]; __shared__ double smax[3][256];
  int t = threadIdx.x;
  for (int b = t; b < nbins_alloc; b += blockDim.x) bin_cnt[b] = 0;
  if (t == 0) { flags[0] = 0; flags[1] = 0; n_edges[0] = 0; }
  if (periodic) return;
  double mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (int i = t; i < N; i += blockDim.x)
    for (int c = 0; c < 3; ++c) { double v = pos[3 * i + c]; mn[c] = fmin(mn[c], v); mx[c] = fmax(mx[c], v); }
  for (int c = 0; c < 3; ++c) { smin[c][t] = mn[c]; smax[c][t] = mx[c]; }
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) for (int c = 0; c < 3; ++c) { smin[c][t] = fmin(smin[c][t], smin[c][t + s]); smax[c][t] = fmax(smax[c][t], smax[c][t + s]); }
    __syncthreads();
  }
  if (t == 0) {
    double d[3];
    for (int c = 0; c < 3; ++c) {
      double lo = smin[c][0], ext = __dsub_rn(smax[c][0], lo);
      // mace: (extent + 2*cutoff + 1) * direction
      d[c] = __dadd_rn(__dadd_rn(ext, 2.0 * cutoff), 1.0);
      int cap = (int)geo[27 + c];
      double bw = fmax(rbin, ext / (double)cap * (1.0 + 1e-9) + 1e-9);
      int nb = (int)floor(ext / bw) + 1; if (nb > cap) nb = cap; if (nb < 1) nb = 1;
      geo[18 + c] = lo; geo[21 + c] = bw; geo[24 + c] = (double)nb;
    }
    const double twopi = 6.283185307179586;  // python 2 * torch.pi
    for (int k = 0; k < 9; ++k) { cellbuf[k] = 0.0; cellbuf[9 + k] = 0.0; }
    for (int c = 0; c < 3; ++c) { cellbuf[4 * c] = d[c]; cellbuf[9 + 4 * c] = __dmul_rn(twopi, __ddiv_rn(1.0, d[c])); }
    cellbuf[18] = __dmul_rn(__dmul_rn(d[0], d[1]), d[2]);
  }
}

// ---- stage 1: per atom: wrap + bin ----
__global__ void k_bin(const double* __restrict__ pos, int N, const double* __restrict__ geo, double* pw, double* w,
                      int* bin_cnt, int* bins, int* atom_bin, int bin_cap, int* flags, int periodic, int ortho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double x[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
  int b[3];
  if (periodic) {
    for (int c = 0; c < 3; ++c) {
      // frac = pos @ inv(cell)  (numpy order (x*i0 + y*i1) + z*i2; exact for orthorhombic)
      double f = __dadd_rn(__dadd_rn(__dmul_rn(x[0], geo[c]), __dmul_rn(x[1], geo[3 + c])), __dmul_rn(x[2], geo[6 + c]));
      double wc = floor(f);
      w[3 * i + c] = wc;
      int nb = (int)geo[24 + c];
      int bb = (int)floor(__dmul_rn(__dsub_rn(f, wc), (double)nb));
      b[c] = bb < 0 ? 0 : (bb >= nb ? nb - 1 : bb);
    }
    // pw = pos - w @ cell with cuBLAS's fp64 [M,3]@[3,3] rounding (fma(w2,C2c, fma(w1,C1c, w0*C0c)), probed bitwise:
    // scripts/nl9/probe_mm2.py) -> identical to proto_graph's torch matmul for orthorhombic AND triclinic cells
    double w0 = w[3 * i], w1 = w[3 * i + 1], w2 = w[3 * i + 2];
    for (int c = 0; c < 3; ++c)
      pw[3 * i + c] = __dsub_rn(x[c], __fma_rn(w2, geo[15 + c], __fma_rn(w1, geo[12 + c], __dmul_rn(w0, geo[9 + c]))));
  } else {
    for (int c = 0; c < 3; ++c) {
      int nb = (int)geo[24 + c];
      int bb = (int)floor(__dsub_rn(x[c], geo[18 + c]) / geo[21 + c]);
      b[c] = bb < 0 ? 0 : (bb >= nb ? nb - 1 : bb);
    }
  }
  int nby = (int)geo[25], nbz = (int)geo[26];
  int bid = (b[0] * nby + b[1]) * nbz + b[2];
  atom_bin[i] = bid;
  int slot = atomicAdd(&bin_cnt[bid], 1);
  if (slot < bin_cap) bins[(long long)bid * bin_cap + slot] = i;
  else atomicOr(&flags[0], 2);
  atomicMax(&flags[1], slot + 1);
}

// pw = pos - wC  (triclinic path; wC from torch matmul exactly like proto)
__global__ void k_sub(const double* __restrict__ pos, const double* __restrict__ wC, double* pw, int n3) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n3) pw[k] = __dsub_rn(pos[k], wC[k]);
}

// ---- neighbour scan for dst atom j: MODE 0 count, MODE 1 write ----
template <int MODE>
__global__ void k_scan(const double* __restrict__ P, int N, const double* __restrict__ geo,
                       const double* __restrict__ ivtab, const int* __restrict__ bin_cnt, const int* __restrict__ bins,
                       const int* __restrict__ atom_bin, int bin_cap, int periodic, double rc2,
                       int* row_cnt, const int* __restrict__ rowptr, long long* src_out, int* img_out, int E_cap) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  int nbx = (int)geo[24], nby = (int)geo[25], nbz = (int)geo[26];
  int bid = atom_bin[j];
  int bz = bid % nbz, by = (bid / nbz) % nby, bx = bid / (nbz * nby);
  double pj0 = P[3 * j], pj1 = P[3 * j + 1], pj2 = P[3 * j + 2];
  int cnt = 0, start = 0, end = 0;
  if (MODE == 1) { start = rowptr[j]; end = rowptr[j + 1]; }
  for (int ox = -1; ox <= 1; ++ox) {
    int rx = bx + ox, tx = 0;
    if (periodic) { if (rx < 0) { rx += nbx; tx = -1; } else if (rx >= nbx) { rx -= nbx; tx = 1; } }
    else if (rx < 0 || rx >= nbx) continue;
    for (int oy = -1; oy <= 1; ++oy) {
      int ry = by + oy, ty = 0;
      if (periodic) { if (ry < 0) { ry += nby; ty = -1; } else if (ry >= nby) { ry -= nby; ty = 1; } }
      else if (ry < 0 || ry >= nby) continue;
      for (int oz = -1; oz <= 1; ++oz) {
        int rz = bz + oz, tz = 0;
        if (periodic) { if (rz < 0) { rz += nbz; tz = -1; } else if (rz >= nbz) { rz -= nbz; tz = 1; } }
        else if (rz < 0 || rz >= nbz) continue;
        // image of sender i near j is pw[i] + t@cell  ->  proto img = -t, d = pw[j] - pw[i] + img@cell
        int ix = -tx, iy = -ty, iz = -tz;
        int m = ((ix + 1) * 3 + (iy + 1)) * 3 + (iz + 1);
        double v0 = ivtab[3 * m], v1 = ivtab[3 * m + 1], v2 = ivtab[3 * m + 2];
        bool zimg = (ix == 0 && iy == 0 && iz == 0);
        int nb = (rx * nby + ry) * nbz + rz;
        int c = bin_cnt[nb]; if (c > bin_cap) c = bin_cap;
        const int* lst = bins + (long long)nb * bin_cap;
        for (int q = 0; q < c; ++q) {
          int i = lst[q];
          if (zimg && i == j) continue;
          double d0 = __dsub_rn(pj0, P[3 * i]), d1 = __dsub_rn(pj1, P[3 * i + 1]), d2 = __dsub_rn(pj2, P[3 * i + 2]);
          if (periodic) { d0 = __dadd_rn(d0, v0); d1 = __dadd_rn(d1, v1); d2 = __dadd_rn(d2, v2); }
          // torch (d*d).sum(-1) over a last dim of 3 reduces as (x^2 + z^2) + y^2 (probed: scripts/nl9/probe_sum.py)
          double r2 = __dadd_rn(__dadd_rn(__dmul_rn(d0, d0), __dmul_rn(d2, d2)), __dmul_rn(d1, d1));
          if (r2 < rc2) {
            if (MODE == 1) {
              int e = start + cnt;
              if (e < end) { src_out[e] = i; img_out[e] = m; }
            }
            ++cnt;
          }
        }
      }
    }
  }
  if (MODE == 0) { row_cnt[j] = cnt; return; }
  // deterministic order: insertion sort row by sender (unique per row: height >= 3 r_max -> one image per pair)
  for (int a = start + 1; a < end; ++a) {
    long long s = src_out[a]; int im = img_out[a]; int b = a - 1;
    while (b >= start && src_out[b] > s) { src_out[b + 1] = src_out[b]; img_out[b + 1] = img_out[b]; --b; }
    src_out[b + 1] = s; img_out[b + 1] = im;
  }
}

// ---- exclusive scan (single block) ----
__global__ void k_rowscan(const int* __restrict__ row_cnt, int N, int* rowptr, int* rowptr_pad, long long* n_edges,
                          int* flags, int E_cap) {
  __shared__ long long part[1024];
  int t = threadIdx.x, T = blockDim.x;
  int chunk = (N + T - 1) / T;
  int lo = t * chunk, hi = min(N, lo + chunk);
  long long s = 0;
  for (int i = lo; i < hi; ++i) s += row_cnt[i];
  part[t] = s;
  __syncthreads();
  if (t == 0) { long long acc = 0; for (int k = 0; k < T; ++k) { long long v = part[k]; part[k] = acc; acc += v; } }
  __syncthreads();
  long long acc = part[t];
  for (int i = lo; i < hi; ++i) {
    long long v = acc < E_cap ? acc : E_cap;
    rowptr[i] = (int)v; rowptr_pad[i] = (int)v;
    acc += row_cnt[i];
  }
  if (hi == N && lo < hi) {
    n_edges[0] = acc;
    rowptr[N] = (int)(acc < E_cap ? acc : E_cap);
    rowptr_pad[N] = E_cap;
    if (acc > E_cap) atomicOr(&flags[0], 1);
  }
  if (N == 0 && t == 0) { rowptr[0] = 0; rowptr_pad[0] = E_cap; }
}


// ---- warp-per-dst-atom neighbour scan (v4). MODE 0: count -> row_cnt. MODE 1: write row in deterministic order.
// The 27 neighbour bins are flattened into one candidate index space (per-warp prefix in shared memory); lanes
// stride over candidates. MODE 1 writes hits unsorted (ballot compaction) into scratch (tmp_src / tmp_img, which
// alias edge_index row 1 and perm_src -- both overwritten later by k_final), then each hit's final slot is its rank
// by sender within the row (senders are unique per row) -> ascending sender order without a sort.
#define WPB 4
template <int MODE>
__global__ void __launch_bounds__(32 * WPB) k_wscan(const double* __restrict__ P, int N, const double* __restrict__ geo,
    const double* __restrict__ ivtab, const int* __restrict__ bin_cnt, const int* __restrict__ bins,
    const int* __restrict__ atom_bin, int bin_cap, int periodic, double rc2, int* row_cnt, const int* __restrict__ rowptr,
    long long* src_out, int* img_out, long long* tmp_src, long long* tmp_img, int E_cap) {
  __shared__ int s_pref[WPB][28]; __shared__ int s_bin[WPB][27]; __shared__ int s_m[WPB][27];
  int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int j = blockIdx.x * WPB + wid;
  if (j >= N) return;   // whole warp exits together
  int nbx = (int)geo[24], nby = (int)geo[25], nbz = (int)geo[26];
  int bid = atom_bin[j];
  int bz = bid % nbz, by = (bid / nbz) % nby, bx = bid / (nbz * nby);
  int c = 0;
  if (lane < 27) {
    int ox = lane / 9 - 1, oy = (lane / 3) % 3 - 1, oz = lane % 3 - 1;
    int rx = bx + ox, ry = by + oy, rz = bz + oz, tx = 0, ty = 0, tz = 0;
    bool valid = true;
    if (periodic) {
      if (rx < 0) { rx += nbx; tx = -1; } else if (rx >= nbx) { rx -= nbx; tx = 1; }
      if (ry < 0) { ry += nby; ty = -1; } else if (ry >= nby) { ry -= nby; ty = 1; }
      if (rz < 0) { rz += nbz; tz = -1; } else if (rz >= nbz) { rz -= nbz; tz = 1; }
    } else valid = rx >= 0 && rx < nbx && ry >= 0 && ry < nby && rz >= 0 && rz < nbz;
    int nb = valid ? (rx * nby + ry) * nbz + rz : 0;
    if (valid) { c = bin_cnt[nb]; if (c > bin_cap) c = bin_cap; }
    s_bin[wid][lane] = nb;
    s_m[wid][lane] = ((-tx + 1) * 3 + (-ty + 1)) * 3 + (-tz + 1);
  }
  // inclusive warp scan of c
  int incl = c;
  for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
  if (lane < 27) s_pref[wid][lane + 1] = incl;
  if (lane == 0) s_pref[wid][0] = 0;
  int T = __shfl_sync(0xffffffffu, incl, 26);
  __syncwarp();
  double pj0 = P[3 * j], pj1 = P[3 * j + 1], pj2 = P[3 * j + 2];
  int start = 0, end = 0;
  if (MODE == 1) { start = rowptr[j]; end = rowptr[j + 1]; }
  int cnt = 0;
  for (int t0 = 0; t0 < T; t0 += 32) {
    int t = t0 + lane;
    bool hit = false; int i = 0, m = 13;
    if (t < T) {
      int k = 0;   // bin slot: largest k with pref[k] <= t
      #pragma unroll
      for (int step = 16; step > 0; step >>= 1) if (k + step <= 26 && s_pref[wid][k + step] <= t) k += step;
      i = bins[(long long)s_bin[wid][k] * bin_cap + (t - s_pref[wid][k])];
      m = s_m[wid][k];
      if (!(m == 13 && i == j)) {
        double d0 = __dsub_rn(pj0, P[3 * i]), d1 = __dsub_rn(pj1, P[3 * i + 1]), d2 = __dsub_rn(pj2, P[3 * i + 2]);
        if (periodic) { d0 = __dadd_rn(d0, ivtab[3 * m]); d1 = __dadd_rn(d1, ivtab[3 * m + 1]); d2 = __dadd_rn(d2, ivtab[3 * m + 2]); }
        // torch (d*d).sum(-1) over a last dim of 3 reduces as (x^2 + z^2) + y^2 (probed: scripts/nl9/probe_sum.py)
        double r2 = __dadd_rn(__dadd_rn(__dmul_rn(d0, d0), __dmul_rn(d2, d2)), __dmul_rn(d1, d1));
        hit = r2 < rc2;
      }
    }
    unsigned bal = __ballot_sync(0xffffffffu, hit);
    if (MODE == 1 && hit) {
      int e = start + cnt + __popc(bal & ((1u << lane) - 1u));
      if (e < end) { tmp_src[e] = i; tmp_img[e] = m; }
    }
    cnt += __popc(bal);
  }
  if (MODE == 0) { if (lane == 0) row_cnt[j] = cnt; return; }
  __syncwarp();
  int L = end - start;
  for (int a = lane; a < L; a += 32) {
    long long sa = tmp_src[start + a];
    int r = 0;
    for (int b = 0; b < L; ++b) r += (tmp_src[start + b] < sa);
    src_out[start + r] = sa; img_out[start + r] = (int)tmp_img[start + a];
    src_out[(long long)E_cap + start + r] = j;   // dst row of edge_index
  }
}

// ---- exclusive scan of row counts (single block, 1024 threads, warp-shuffle block scan) ----
__global__ void k_rowscan2(const int* __restrict__ row_cnt, int N, int* rowptr, int* rowptr_pad, long long* n_edges,
                           int* flags, int E_cap) {
  __shared__ long long wsum[32];
  int t = threadIdx.x, lane = t & 31, wid = t >> 5;
  int chunk = (N + 1023) / 1024;
  int lo = min(N, t * chunk), hi = min(N, lo + chunk);
  long long s = 0;
  for (int i = lo; i < hi; ++i) s += row_cnt[i];
  long long incl = s;
  for (int o = 1; o < 32; o <<= 1) { long long v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
  if (lane == 31) wsum[wid] = incl;
  __syncthreads();
  if (wid == 0) {
    long long v = wsum[lane], x = v;
    for (int o = 1; o < 32; o <<= 1) { long long u = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += u; }
    wsum[lane] = x - v;   // exclusive warp offsets
  }
  __syncthreads();
  long long acc = wsum[wid] + incl - s;
  for (int i = lo; i < hi; ++i) {
    long long v = acc < E_cap ? acc : E_cap;
    rowptr[i] = (int)v; rowptr_pad[i] = (int)v;
    acc += row_cnt[i];
  }
  if (t == 1023) {
    n_edges[0] = acc;   // thread 1023 ends at N (lo=hi=N if idle) and acc == total
    rowptr[N] = (int)(acc < E_cap ? acc : E_cap);
    rowptr_pad[N] = E_cap;
    if (acc > E_cap) atomicOr(&flags[0], 1);
  }
}

// ---- finalize: dst index, unit shifts, shifts, reverse map -> perm_src, padding ----
__global__ void k_final(int N, const int* __restrict__ rowptr, const long long* __restrict__ n_edges,
                        const double* __restrict__ w, const double* __restrict__ cellv, long long* ei, const int* __restrict__ img,
                        double* us, double* sh, long long* perm_src, int* flags, int E_cap, int periodic, int pad_m) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E_cap) return;
  long long E = n_edges[0];
  int Ef = rowptr[N];
  double C[9];
  for (int k = 0; k < 9; ++k) C[k] = cellv[k];
  if (e >= Ef) {   // padding
    ei[e] = N - 1; ei[E_cap + e] = N - 1;
    double m = (double)pad_m;
    us[3 * e] = m; us[3 * e + 1] = 0.0; us[3 * e + 2] = 0.0;
    for (int c = 0; c < 3; ++c)
      sh[3 * e + c] = __fma_rn(0.0, C[6 + c], __fma_rn(0.0, C[3 + c], __dmul_rn(m, C[c])));
    perm_src[e] = e;
    return;
  }
  long long j = ei[E_cap + e];   // dst written by k_wscan<1>
  long long i = ei[e];
  double S[3];
  if (periodic) {
    int m = img[e];
    double im[3] = {(double)(m / 9 - 1), (double)((m / 3) % 3 - 1), (double)(m % 3 - 1)};
    for (int c = 0; c < 3; ++c) S[c] = __dadd_rn(__dsub_rn(im[c], w[3 * j + c]), w[3 * i + c]);
  } else { S[0] = S[1] = S[2] = 0.0; }
  for (int c = 0; c < 3; ++c) {
    us[3 * e + c] = S[c];
    // shifts = S @ cell with cuBLAS's rounding (bitwise equal to proto_graph's torch matmul)
    sh[3 * e + c] = __fma_rn(S[2], C[6 + c], __fma_rn(S[1], C[3 + c], __dmul_rn(S[0], C[c])));
  }
  if (E > E_cap) { perm_src[e] = e; return; }   // overflow: outputs invalid anyway; stay memory-safe
  // reverse edge (j -> i, -S) lives in row i at the slot whose sender == j
  int a = rowptr[i], b = rowptr[i + 1] - 1, p = -1;
  while (a <= b) { int mid = (a + b) >> 1; long long s = ei[mid]; if (s == j) { p = mid; break; } if (s < j) a = mid + 1; else b = mid - 1; }
  if (p < 0) { atomicOr(&flags[0], 4); p = e; }
  perm_src[e] = p;
}

void nl_build(torch::Tensor pos, torch::Tensor pw, torch::Tensor w, torch::Tensor geo, torch::Tensor ivtab,
              torch::Tensor bin_cnt, torch::Tensor bins, torch::Tensor atom_bin, torch::Tensor row_cnt,
              torch::Tensor rowptr, torch::Tensor rowptr_pad, torch::Tensor edge_index, torch::Tensor img_pack,
              torch::Tensor unit_shifts, torch::Tensor shifts, torch::Tensor perm_src, torch::Tensor n_edges,
              torch::Tensor flags, torch::Tensor cellbuf, torch::Tensor pw_mm, torch::Tensor wC,
              int64_t periodic, int64_t ortho, int64_t bin_cap, int64_t nb_cap, double rc2, double cutoff, double rbin,
              int64_t pad_m, int64_t stage) {
  auto st = at::cuda::getCurrentCUDAStream();
  int N = pos.size(0);
  int E_cap = shifts.size(0);
  const int TB = 128;
  if (stage == 0) {
    k_prep<<<1, 256, 0, st>>>(pos.data_ptr<double>(), N, geo.data_ptr<double>(), bin_cnt.data_ptr<int>(), (int)nb_cap,
                             flags.data_ptr<int>(), (long long*)n_edges.data_ptr<int64_t>(), cellbuf.data_ptr<double>(),
                             (int)periodic, cutoff, rbin);
    k_bin<<<(N + TB - 1) / TB, TB, 0, st>>>(pos.data_ptr<double>(), N, geo.data_ptr<double>(), pw.data_ptr<double>(),
                             w.data_ptr<double>(), bin_cnt.data_ptr<int>(), bins.data_ptr<int>(), atom_bin.data_ptr<int>(),
                             (int)bin_cap, flags.data_ptr<int>(), (int)periodic, (int)ortho);
    return;
  }
  if (stage == 1) {   // triclinic: pw = pos - wC (wC computed by torch.matmul in python)
    k_sub<<<(3 * N + 255) / 256, 256, 0, st>>>(pos.data_ptr<double>(), wC.data_ptr<double>(), pw.data_ptr<double>(), 3 * N);
    return;
  }
  const double* P = periodic ? pw.data_ptr<double>() : pos.data_ptr<double>();
  long long* ei = (long long*)edge_index.data_ptr<int64_t>();
  // scratch for unsorted hits: unit_shifts storage (>= 3*E_cap*8 bytes), fully rewritten by k_final afterwards
  long long* tmp_src = (long long*)unit_shifts.data_ptr<double>();
  long long* tmp_img = tmp_src + E_cap;
  int nblk = (N + WPB - 1) / WPB;
  k_wscan<0><<<nblk, 32 * WPB, 0, st>>>(P, N, geo.data_ptr<double>(), ivtab.data_ptr<double>(), bin_cnt.data_ptr<int>(),
      bins.data_ptr<int>(), atom_bin.data_ptr<int>(), (int)bin_cap, (int)periodic, rc2, row_cnt.data_ptr<int>(),
      rowptr.data_ptr<int>(), ei, img_pack.data_ptr<int>(), tmp_src, tmp_img, E_cap);
  k_rowscan2<<<1, 1024, 0, st>>>(row_cnt.data_ptr<int>(), N, rowptr.data_ptr<int>(), rowptr_pad.data_ptr<int>(),
      (long long*)n_edges.data_ptr<int64_t>(), flags.data_ptr<int>(), E_cap);
  k_wscan<1><<<nblk, 32 * WPB, 0, st>>>(P, N, geo.data_ptr<double>(), ivtab.data_ptr<double>(), bin_cnt.data_ptr<int>(),
      bins.data_ptr<int>(), atom_bin.data_ptr<int>(), (int)bin_cap, (int)periodic, rc2, row_cnt.data_ptr<int>(),
      rowptr.data_ptr<int>(), ei, img_pack.data_ptr<int>(), tmp_src, tmp_img, E_cap);
  if (E_cap > 0)
    k_final<<<(E_cap + 255) / 256, 256, 0, st>>>(N, rowptr.data_ptr<int>(), (const long long*)n_edges.data_ptr<int64_t>(),
        w.data_ptr<double>(), cellbuf.data_ptr<double>(), ei, img_pack.data_ptr<int>(), unit_shifts.data_ptr<double>(),
        shifts.data_ptr<double>(), (long long*)perm_src.data_ptr<int64_t>(), flags.data_ptr<int>(), E_cap, (int)periodic,
        (int)pad_m);
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(name="nl9_cell_v5", cpp_sources=_CPP, cuda_sources=_CU, functions=["nl_build"],
                           extra_cuda_cflags=["-O3", "--fmad=false", "-gencode=arch=compute_90,code=sm_90"],
                           verbose=False)
    return _EXT


def round_cap(x, mult=64):
    return max(mult, int(math.ceil(x / mult)) * mult)


class CellNeighborList:
    BIN_MARGIN = 1e-9   # bins are >= r_max*(1+1e-9): rounding in bin assignment can never hide a pair

    def __init__(self, r_max, pbc, cell, n_atoms, E_cap, device="cuda", bin_cap=None, bins_per_dim=None):
        self.r_max = float(r_max)
        self.cutoff = self.r_max
        self.rc2 = self.cutoff * self.cutoff          # same python fp64 product as proto_graph
        pbc = tuple(bool(x) for x in (pbc if np.ndim(pbc) else (pbc,) * 3))
        if any(pbc) and not all(pbc):
            raise NotImplementedError("mixed pbc not supported (use proto_graph)")
        self.pbc = pbc
        self.periodic = all(pbc)
        self.N = int(n_atoms)
        if self.N < 1:
            raise NotImplementedError("n_atoms < 1")
        self.E_cap = int(E_cap)
        self.dev = torch.device(device)
        rbin = self.r_max * (1.0 + self.BIN_MARGIN)
        self.rbin = rbin
        geo = np.zeros(31, dtype=np.float64)
        cellbuf = np.zeros(19, dtype=np.float64)
        dev = self.dev
        if self.periodic:
            cell = np.array(cell, dtype=np.float64).reshape(3, 3)
            if abs(np.linalg.det(cell)) < 1e-12:
                raise NotImplementedError("degenerate periodic cell")
            inv = np.linalg.inv(cell)                                   # same as proto (numpy)
            heights = 1.0 / np.linalg.norm(inv, axis=0)
            nb = [int(math.floor(h / rbin)) for h in heights]
            if min(nb) < 3:
                raise NotImplementedError(f"periodic cell too small for cell list (bins per dim {nb}, need >= 3)")
            self.nb = nb
            self.ortho = bool(np.count_nonzero(cell - np.diag(np.diag(cell))) == 0)
            geo[0:9] = inv.reshape(-1); geo[9:18] = cell.reshape(-1); geo[24:27] = nb
            nbins = nb[0] * nb[1] * nb[2]
            # returned cell / rcell / volume exactly as proto_graph (CPU torch)
            cell_t = torch.tensor(cell, dtype=torch.float64)
            volume = torch.linalg.det(cell_t)
            rcell = 2 * torch.pi * torch.linalg.inv(cell_t.mT)
            cellbuf[0:9] = cell_t.numpy().reshape(-1); cellbuf[9:18] = rcell.numpy().reshape(-1); cellbuf[18] = volume.item()
            # 27 image vectors: same torch op/shape as proto (img [27,3] @ search_cell on GPU)
            r = [torch.arange(-1, 2, dtype=torch.float64)] * 3
            img = torch.cartesian_prod(*r).reshape(-1, 3).to(dev)
            self.ivtab = (img @ torch.tensor(cell, dtype=torch.float64).to(dev)).contiguous()
            self.cell_dev = torch.tensor(cell, dtype=torch.float64, device=dev)
            self.pad_m = max(1, int(math.ceil(2 * self.r_max / float(np.linalg.norm(cell[0])))))
            bin_vol = abs(np.linalg.det(cell)) / nbins
            dflt_cap = max(64, round_cap(bin_vol * 0.25, 32))
        else:
            self.ortho = True
            if bins_per_dim is None:
                bins_per_dim = max(4, min(64, int(math.ceil(self.N ** (1 / 3) * 1.2)) + 2))
            bpd = [int(bins_per_dim)] * 3 if np.ndim(bins_per_dim) == 0 else [int(x) for x in bins_per_dim]
            geo[27:30] = bpd
            nbins = bpd[0] * bpd[1] * bpd[2]
            self.nb = bpd
            self.ivtab = torch.zeros((27, 3), dtype=torch.float64, device=dev)
            self.pad_m = 1
            dflt_cap = 64
        self.bin_cap = int(bin_cap) if bin_cap is not None else dflt_cap
        self.nbins = nbins
        N, Ec = self.N, self.E_cap
        i32, i64, f64 = torch.int32, torch.int64, torch.float64
        self.geo = torch.tensor(geo, device=dev)
        self.cellbuf = torch.tensor(cellbuf, device=dev)
        self.pw = torch.zeros((N, 3), dtype=f64, device=dev)
        self.w = torch.zeros((N, 3), dtype=f64, device=dev)
        self.wC = torch.zeros(1, dtype=f64, device=dev)   # unused (legacy stage-1 arg)
        self.bin_cnt = torch.zeros(nbins, dtype=i32, device=dev)
        self.bins = torch.zeros(nbins * self.bin_cap, dtype=i32, device=dev)
        self.atom_bin = torch.zeros(N, dtype=i32, device=dev)
        self.row_cnt = torch.zeros(N, dtype=i32, device=dev)
        self.rowptr = torch.zeros(N + 1, dtype=i32, device=dev)
        self.rowptr_pad = torch.zeros(N + 1, dtype=i32, device=dev)
        self.edge_index = torch.zeros((2, Ec), dtype=i64, device=dev)
        self.img = torch.zeros(Ec, dtype=i32, device=dev)
        self.unit_shifts = torch.zeros((Ec, 3), dtype=f64, device=dev)
        self.shifts = torch.zeros((Ec, 3), dtype=f64, device=dev)
        self.perm_src = torch.zeros(Ec, dtype=i64, device=dev)
        self.perm_dst = torch.arange(Ec, dtype=i64, device=dev)
        self.n_edges = torch.zeros((), dtype=i64, device=dev)
        self.flags = torch.zeros(2, dtype=i32, device=dev)           # [overflow bits, max atoms in a bin]
        self.cell_v = self.cellbuf[0:9].view(3, 3)
        self.rcell_v = self.cellbuf[9:18].view(3, 3)
        self.volume_v = self.cellbuf[18:19]
        self.out = dict(edge_index=self.edge_index, unit_shifts=self.unit_shifts, shifts=self.shifts,
                        rowptr_dst=self.rowptr, rowptr_src=self.rowptr, rowptr_dst_pad=self.rowptr_pad,
                        rowptr_src_pad=self.rowptr_pad, perm_dst=self.perm_dst, perm_src=self.perm_src,
                        n_edges=self.n_edges, overflow=self.flags[0], max_bin=self.flags[1],
                        cell=self.cell_v, rcell=self.rcell_v, volume=self.volume_v)
        _ext()

    def memory_bytes(self):
        ts = [self.geo, self.cellbuf, self.pw, self.w, self.wC, self.bin_cnt, self.bins, self.atom_bin, self.row_cnt,
              self.rowptr, self.rowptr_pad, self.edge_index, self.img, self.unit_shifts, self.shifts, self.perm_src,
              self.perm_dst, self.n_edges, self.flags, self.ivtab]
        return int(sum(t.numel() * t.element_size() for t in ts))

    def build(self, positions):
        """positions [N,3] fp64 contiguous on device. No host sync; CUDA-graph capturable. Returns self.out
        (persistent buffers). Caller must check out['overflow'] (0 = valid)."""
        if positions.shape != (self.N, 3) or positions.dtype != torch.float64 or positions.device != self.pw.device:
            raise ValueError("positions must be [N,3] float64 on the builder device")
        pos = positions if positions.is_contiguous() else positions.contiguous()
        ext = _ext()
        args = [pos, self.pw, self.w, self.geo, self.ivtab, self.bin_cnt, self.bins, self.atom_bin, self.row_cnt,
                self.rowptr, self.rowptr_pad, self.edge_index, self.img, self.unit_shifts, self.shifts, self.perm_src,
                self.n_edges, self.flags, self.cellbuf, self.pw, self.wC,
                int(self.periodic), int(self.ortho), self.bin_cap, self.nbins, self.rc2, self.cutoff, self.rbin, self.pad_m]
        ext.nl_build(*args, 0)
        ext.nl_build(*args, 2)
        return self.out


class BatchTemplate:
    """Static part of the MACE batch dict for one configuration key (numbers, pbc, head), taken once from a stock batch
    exactly like proto_graph.ProtoBuilder._template."""
    DYN = ("positions", "edge_index", "shifts", "unit_shifts", "cell", "rcell", "volume")

    def __init__(self, calc, atoms):
        import sys, os
        sys.path.insert(0, os.path.join(os.environ.get("R", "."), "scripts/graph"))
        import proto_graph
        pb = proto_graph.ProtoBuilder(calc)
        pb._check_supported(atoms)
        t = pb._template(atoms)
        self.keys = t["keys"]
        self.static = t["static"]
        self.pbc = t["pbc"]

    def batch(self, out, positions, trim=False):
        if trim:
            E = int(out["n_edges"].item())                   # host sync (testing / eager use only)
            ei, us, sh = out["edge_index"][:, :E], out["unit_shifts"][:E], out["shifts"][:E]
        else:
            ei, us, sh = out["edge_index"], out["unit_shifts"], out["shifts"]
        dyn = dict(positions=positions, edge_index=ei, shifts=sh, unit_shifts=us,
                   cell=out["cell"], rcell=out["rcell"], volume=out["volume"])
        return {k: (dyn[k] if k in dyn else self.static[k]) for k in self.keys}


def batch_dict(template, out, positions, trim=False):
    return template.batch(out, positions, trim=trim)
