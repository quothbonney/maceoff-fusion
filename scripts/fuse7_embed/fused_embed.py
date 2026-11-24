"""fuse7_embed: fused fp64 edge-embedding kernels for MACE (0.3.17) ScaleShiftMACE / MACE.

Replaces (in-memory module swap, no MACE source edits):
  model.spherical_harmonics  (e3nn o3.SphericalHarmonics lmax=3, normalize=True, normalization='component')
      -> FusedSphericalHarmonics: 1 CUDA kernel fwd (fe_sh_fwd), 1 CUDA kernel bwd (fe_sh_bwd)
  model.radial_embedding     (RadialEmbeddingBlock: BesselBasis x PolynomialCutoff, no distance transform)
      -> FusedRadialEmbedding:  1 CUDA kernel fwd (fe_radial_fwd), 1 CUDA kernel bwd (fe_radial_bwd)

Inputs are exactly what ScaleShiftMACE.forward passes (vectors [E,3], lengths [E,1]), so gradients flow back to
positions / displacement (stress) through MACE's own get_edge_vectors_and_lengths autograd graph. Two independent
modules -> no cross-module cache/handoff, so nothing can go stale across calls or CUDA-graph replays.

Backward: custom first derivative (torch.autograd.Function + once_differentiable -> double backward raises).
CUDA-graph safe: no host syncs, outputs allocated with shapes depending only on E, launches on torch's current
stream, E == 0 skips the launch.

API: install(model) -> handle, uninstall(model, handle), describe() -> dict
"""
import math
import os

import torch
from torch.autograd.function import once_differentiable

_EXT = None

_CPP = r"""
#include <torch/extension.h>
torch::Tensor sh_fwd(torch::Tensor v, double eps, torch::Tensor cf);
torch::Tensor sh_bwd(torch::Tensor v, torch::Tensor g, double eps, torch::Tensor cf);
torch::Tensor radial_fwd(torch::Tensor r, torch::Tensor w, double pref, double rmax, double p);
torch::Tensor radial_bwd(torch::Tensor r, torch::Tensor g, torch::Tensor w, double pref, double rmax, double p);
"""

_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define S3 1.7320508075688772
#define CA 0.9128709291752769   /* sqrt(5/6) */
#define CB 2.23606797749979     /* sqrt(5)   */
#define CC 0.6123724356957945   /* sqrt(3/8) */

// unit vector exactly like torch.nn.functional.normalize: x / clamp_min(||x||, eps)
__device__ __forceinline__ double fe_norm(double x, double y, double z) {
  return __dsqrt_rn(__dadd_rn(__dadd_rn(__dmul_rn(x, x), __dmul_rn(y, y)), __dmul_rn(z, z)));
}

__global__ void fe_sh_fwd(const double* __restrict__ v, double* __restrict__ out, long E, double eps,
                          const double* __restrict__ cf) {
  long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  double vx = v[3*e], vy = v[3*e+1], vz = v[3*e+2];
  double n = fe_norm(vx, vy, vz); double d = n > eps ? n : eps;
  double x = vx / d, y = vy / d, z = vz / d;
  double sh20 = S3 * x * z, sh21 = S3 * x * y, y2 = y * y, x2z2 = x * x + z * z;
  double sh22 = y2 - 0.5 * x2z2, sh23 = S3 * y * z, sh24 = S3 / 2.0 * (z * z - x * x);
  double o[16];
  o[0] = 1.0; o[1] = x; o[2] = y; o[3] = z;
  o[4] = sh20; o[5] = sh21; o[6] = sh22; o[7] = sh23; o[8] = sh24;
  o[9]  = CA * (sh20 * z + sh24 * x);
  o[10] = CB * sh20 * y;
  o[11] = CC * (4.0 * y2 - x2z2) * x;
  o[12] = 0.5 * y * (2.0 * y2 - 3.0 * x2z2);
  o[13] = CC * z * (4.0 * y2 - x2z2);
  o[14] = CB * sh24 * y;
  o[15] = CA * (sh24 * z - sh20 * x);
  double* op = out + 16 * e;
#pragma unroll
  for (int k = 0; k < 16; ++k) op[k] = o[k] * cf[k];
}

__global__ void fe_sh_bwd(const double* __restrict__ v, const double* __restrict__ gin, double* __restrict__ gv,
                          long E, double eps, const double* __restrict__ cf) {
  long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  double vx = v[3*e], vy = v[3*e+1], vz = v[3*e+2];
  double n = fe_norm(vx, vy, vz); double d = n > eps ? n : eps;
  double x = vx / d, y = vy / d, z = vz / d;
  double G[16];
  const double* gp = gin + 16 * e;
#pragma unroll
  for (int k = 0; k < 16; ++k) G[k] = gp[k] * cf[k];
  double sh20 = S3 * x * z, y2 = y * y, x2z2 = x * x + z * z, sh24 = S3 / 2.0 * (z * z - x * x);
  double t = 4.0 * y2 - x2z2;
  double gx = G[1], gy = G[2], gz = G[3];
  double g20 = G[4], g21 = G[5], g22 = G[6], g23 = G[7], g24 = G[8], gy2 = 0.0, gx2z2 = 0.0, gt;
  // l = 3
  g20 += CA * G[9] * z; g24 += CA * G[9] * x; gz += CA * G[9] * sh20; gx += CA * G[9] * sh24;
  g20 += CB * G[10] * y; gy += CB * G[10] * sh20;
  gt = CC * G[11] * x; gx += CC * G[11] * t; gy2 += 4.0 * gt; gx2z2 -= gt;
  gy += 0.5 * G[12] * (2.0 * y2 - 3.0 * x2z2); gy2 += G[12] * y; gx2z2 -= 1.5 * G[12] * y;
  gt = CC * G[13] * z; gz += CC * G[13] * t; gy2 += 4.0 * gt; gx2z2 -= gt;
  g24 += CB * G[14] * y; gy += CB * G[14] * sh24;
  g24 += CA * G[15] * z; gz += CA * G[15] * sh24; g20 -= CA * G[15] * x; gx -= CA * G[15] * sh20;
  // l = 2
  gy2 += g22; gx2z2 -= 0.5 * g22;
  gx += S3 * z * g20; gz += S3 * x * g20;
  gx += S3 * y * g21; gy += S3 * x * g21;
  gy += S3 * z * g23; gz += S3 * y * g23;
  gz += S3 * z * g24; gx -= S3 * x * g24;
  gy += 2.0 * y * gy2; gx += 2.0 * x * gx2z2; gz += 2.0 * z * gx2z2;
  double* o = gv + 3 * e;
  if (n > eps) {   // d(v/|v|)/dv = (I - u u^T)/|v|
    double ug = x * gx + y * gy + z * gz;
    o[0] = (gx - x * ug) / n; o[1] = (gy - y * ug) / n; o[2] = (gz - z * ug) / n;
  } else {         // clamp_min active: v/eps
    o[0] = gx / eps; o[1] = gy / eps; o[2] = gz / eps;
  }
}

// MACE RadialEmbeddingBlock: prefactor * sin(w_k r)/r * envelope(r) * (r < rmax)
#define MAXB 32
__global__ void fe_radial_fwd(const double* __restrict__ r, const double* __restrict__ w, double* __restrict__ out,
                              long E, int B, double pref, double rmax, double p) {
  long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  double x = r[e];
  double u = x / rmax;
  double env = 1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * pow(u, p) + p * (p + 2.0) * pow(u, p + 1.0)
               - (p * (p + 1.0) / 2.0) * pow(u, p + 2.0);
  env = env * (x < rmax ? 1.0 : 0.0);
  double* op = out + (long)B * e;
  for (int k = 0; k < B; ++k) op[k] = (pref * (sin(w[k] * x) / x)) * env;
}

__global__ void fe_radial_bwd(const double* __restrict__ r, const double* __restrict__ gin, const double* __restrict__ w,
                              double* __restrict__ gr, long E, int B, double pref, double rmax, double p) {
  long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  double x = r[e];
  double m = x < rmax ? 1.0 : 0.0;
  double u = x / rmax;
  double env = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * pow(u, p) + p * (p + 2.0) * pow(u, p + 1.0)
               - (p * (p + 1.0) / 2.0) * pow(u, p + 2.0)) * m;
  double denv = (-((p + 1.0) * (p + 2.0) / 2.0) * p * pow(u, p - 1.0) + p * (p + 2.0) * (p + 1.0) * pow(u, p)
                 - (p * (p + 1.0) / 2.0) * (p + 2.0) * pow(u, p + 1.0)) / rmax * m;
  const double* gp = gin + (long)B * e;
  double acc_rad = 0.0, acc_env = 0.0;
  for (int k = 0; k < B; ++k) {
    double s, c; sincos(w[k] * x, &s, &c);
    double rad = pref * (s / x);
    double drad = pref * (w[k] * c * x - s) / (x * x);
    acc_env += gp[k] * rad;
    acc_rad += gp[k] * drad;
  }
  gr[e] = acc_rad * env + acc_env * denv;
}

static inline void fe_check(const torch::Tensor& t, const char* nm) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kDouble && t.is_contiguous(), nm, " must be contiguous cuda fp64");
}
static inline unsigned fe_grid(long E) { return (unsigned)((E + 127) / 128); }

torch::Tensor sh_fwd(torch::Tensor v, double eps, torch::Tensor cf) {
  fe_check(v, "vectors"); fe_check(cf, "cf");
  c10::cuda::CUDAGuard g(v.device());
  long E = v.size(0);
  auto out = torch::empty({E, 16}, v.options());
  if (E > 0) fe_sh_fwd<<<fe_grid(E), 128, 0, at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<double>(), out.data_ptr<double>(), E, eps, cf.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
torch::Tensor sh_bwd(torch::Tensor v, torch::Tensor gin, double eps, torch::Tensor cf) {
  fe_check(v, "vectors"); fe_check(gin, "grad"); fe_check(cf, "cf");
  c10::cuda::CUDAGuard g(v.device());
  long E = v.size(0);
  auto gv = torch::empty({E, 3}, v.options());
  if (E > 0) fe_sh_bwd<<<fe_grid(E), 128, 0, at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<double>(), gin.data_ptr<double>(), gv.data_ptr<double>(), E, eps, cf.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gv;
}
torch::Tensor radial_fwd(torch::Tensor r, torch::Tensor w, double pref, double rmax, double p) {
  fe_check(r, "lengths"); fe_check(w, "bessel_weights");
  c10::cuda::CUDAGuard g(r.device());
  long E = r.size(0); int B = (int)w.numel();
  TORCH_CHECK(B <= MAXB);
  auto out = torch::empty({E, B}, r.options());
  if (E > 0) fe_radial_fwd<<<fe_grid(E), 128, 0, at::cuda::getCurrentCUDAStream()>>>(r.data_ptr<double>(), w.data_ptr<double>(), out.data_ptr<double>(), E, B, pref, rmax, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
torch::Tensor radial_bwd(torch::Tensor r, torch::Tensor gin, torch::Tensor w, double pref, double rmax, double p) {
  fe_check(r, "lengths"); fe_check(gin, "grad"); fe_check(w, "bessel_weights");
  c10::cuda::CUDAGuard g(r.device());
  long E = r.size(0); int B = (int)w.numel();
  auto gr = torch::empty({E, 1}, r.options());
  if (E > 0) fe_radial_bwd<<<fe_grid(E), 128, 0, at::cuda::getCurrentCUDAStream()>>>(r.data_ptr<double>(), gin.data_ptr<double>(), w.data_ptr<double>(), gr.data_ptr<double>(), E, B, pref, rmax, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gr;
}
"""


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(name="fuse7_embed_v1", cpp_sources=_CPP, cuda_sources=_CU,
                           functions=["sh_fwd", "sh_bwd", "radial_fwd", "radial_bwd"],
                           extra_cuda_cflags=["-O3", "--fmad=false", "-gencode=arch=compute_90,code=sm_90"],
                           verbose=False)
    return _EXT


SH_EPS = 1e-12  # torch.nn.functional.normalize default


class _SHFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, cf):
        v = v.contiguous()
        ctx.save_for_backward(v, cf)
        return ext().sh_fwd(v, SH_EPS, cf)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        v, cf = ctx.saved_tensors
        return ext().sh_bwd(v, g.contiguous(), SH_EPS, cf), None


class _RadialFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, pref, rmax, p):
        r = r.contiguous()
        ctx.save_for_backward(r, w)
        ctx.c = (pref, rmax, p)
        return ext().radial_fwd(r, w, pref, rmax, p)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        r, w = ctx.saved_tensors
        return ext().radial_bwd(r, g.contiguous(), w, *ctx.c), None, None, None, None


class FusedSphericalHarmonics(torch.nn.Module):
    """Drop-in for e3nn o3.SphericalHarmonics(lmax=3, normalize=True, normalization='component')."""

    def __init__(self, orig):
        super().__init__()
        self.irreps_out = orig.irreps_out
        self.irreps_in = getattr(orig, "irreps_in", None)
        cf = [math.sqrt(2 * l + 1) for l in range(4) for _ in range(2 * l + 1)]
        self.register_buffer("cf", torch.tensor(cf, dtype=torch.float64), persistent=False)

    def forward(self, x):
        if x.dtype != torch.float64 or not x.is_cuda:
            raise RuntimeError("fuse7_embed: FusedSphericalHarmonics requires CUDA fp64 input")
        if self.cf.device != x.device:
            raise RuntimeError("fuse7_embed: cf buffer on wrong device (move model before install)")
        return _SHFn.apply(x, self.cf)


class FusedRadialEmbedding(torch.nn.Module):
    """Drop-in for MACE RadialEmbeddingBlock(bessel, PolynomialCutoff, no distance transform, apply_cutoff=True)."""

    def __init__(self, orig):
        super().__init__()
        bf, cut = orig.bessel_fn, orig.cutoff_fn
        self.out_dim = orig.out_dim
        self.register_buffer("w", bf.bessel_weights.detach().clone().to(torch.float64).contiguous(), persistent=False)
        # host scalars read once at install (not per call -> no sync during capture)
        self.pref = float(bf.prefactor)
        self.rmax = float(cut.r_max)
        self.p = float(int(cut.p))
        if abs(float(bf.r_max) - self.rmax) > 0:
            raise RuntimeError("fuse7_embed: bessel r_max != cutoff r_max")

    def forward(self, edge_lengths, node_attrs=None, edge_index=None, atomic_numbers=None):
        if edge_lengths.dtype != torch.float64 or not edge_lengths.is_cuda:
            raise RuntimeError("fuse7_embed: FusedRadialEmbedding requires CUDA fp64 input")
        return _RadialFn.apply(edge_lengths, self.w, self.pref, self.rmax, self.p), None


def _check_supported(model):
    from e3nn import o3
    from mace.modules.blocks import RadialEmbeddingBlock
    from mace.modules.radial import BesselBasis, PolynomialCutoff
    sh = getattr(model, "spherical_harmonics", None)
    re = getattr(model, "radial_embedding", None)
    errs = []
    if not isinstance(sh, o3.SphericalHarmonics):
        errs.append(f"spherical_harmonics is {type(sh)}")
    else:
        if sh._lmax != 3 or not sh._is_range_lmax: errs.append(f"sh lmax={sh._lmax} range={sh._is_range_lmax}")
        if not sh.normalize: errs.append("sh normalize=False")
        if sh.normalization != "component": errs.append(f"sh normalization={sh.normalization}")
        if str(sh.irreps_out) != str(o3.Irreps.spherical_harmonics(3)): errs.append(f"sh irreps_out={sh.irreps_out}")
    if not isinstance(re, RadialEmbeddingBlock):
        errs.append(f"radial_embedding is {type(re)}")
    else:
        if type(re.bessel_fn) is not BesselBasis: errs.append(f"radial basis {type(re.bessel_fn)}")
        elif re.bessel_fn.bessel_weights.requires_grad: errs.append("trainable bessel weights (grad not provided)")
        if type(re.cutoff_fn) is not PolynomialCutoff: errs.append(f"cutoff {type(re.cutoff_fn)}")
        if hasattr(re, "distance_transform"): errs.append(f"distance_transform {type(re.distance_transform)}")
        if hasattr(re, "apply_cutoff") and not re.apply_cutoff: errs.append("apply_cutoff=False")
        if len(re.bessel_fn.bessel_weights) > 32: errs.append("num_bessel > 32")
    if errs:
        raise NotImplementedError("fuse7_embed: unsupported model: " + "; ".join(errs))
    if sh is not None:
        pass


def install(model):
    """Swap model.spherical_harmonics and model.radial_embedding for fused-kernel modules. Returns handle."""
    if getattr(model, "_fuse7_embed_installed", False):
        raise RuntimeError("fuse7_embed: already installed on this model")
    _check_supported(model)
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError("fuse7_embed: model must be on CUDA before install")
    ext()  # build/load now, not inside a capture
    orig_sh, orig_re = model.spherical_harmonics, model.radial_embedding
    new_sh = FusedSphericalHarmonics(orig_sh).to(dev)
    new_re = FusedRadialEmbedding(orig_re).to(dev)
    model.spherical_harmonics = new_sh
    model.radial_embedding = new_re
    model._fuse7_embed_installed = True
    return {"spherical_harmonics": orig_sh, "radial_embedding": orig_re, "new": (new_sh, new_re)}


def uninstall(model, handle):
    if handle is None or not getattr(model, "_fuse7_embed_installed", False):
        return
    model.spherical_harmonics = handle["spherical_harmonics"]
    model.radial_embedding = handle["radial_embedding"]
    model._fuse7_embed_installed = False


def describe():
    return {
        "replaces": {
            "model.spherical_harmonics": "e3nn o3.SphericalHarmonics(lmax=3, normalize=True, normalization='component')",
            "model.radial_embedding": "mace RadialEmbeddingBlock(BesselBasis x PolynomialCutoff), returns (edge_feats, None)",
        },
        "kernels": {"fwd": ["fe_sh_fwd", "fe_radial_fwd"], "bwd": ["fe_sh_bwd", "fe_radial_bwd"]},
        "launches_per_forward": 2, "launches_per_backward": 2,
        "grad_wrt": {"spherical_harmonics": "vectors [E,3]", "radial_embedding": "lengths [E,1]"},
        "not_fused": "get_edge_vectors_and_lengths (gather, sub, norm) stays stock MACE autograd",
        "double_backward": "raises (once_differentiable)",
        "precision": "fp64, --fmad=false",
        "cuda_graph": "capturable: current stream, no host sync, E==0 skips launch",
    }
