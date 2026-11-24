"""Probe: which explicit FMA chain reproduces cuBLAS fp64 [M,3]@[3,3] bitwise (for in-kernel shifts / pw)."""
import os, torch
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
__global__ void k(const double* A, const double* C, double* out, int M, int v) {
  int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= M) return;
  for (int c = 0; c < 3; ++c) {
    double a0 = A[3*r], a1 = A[3*r+1], a2 = A[3*r+2], o;
    if (v == 0) o = __fma_rn(a2, C[6+c], __fma_rn(a1, C[3+c], __dmul_rn(a0, C[c])));
    else if (v == 1) o = __fma_rn(a0, C[c], __fma_rn(a1, C[3+c], __dmul_rn(a2, C[6+c])));
    else if (v == 2) o = __fma_rn(a1, C[3+c], __fma_rn(a0, C[c], __dmul_rn(a2, C[6+c])));
    else o = __fma_rn(a2, C[6+c], __fma_rn(a0, C[c], __dmul_rn(a1, C[3+c])));
    out[3*r+c] = o;
  }
}
torch::Tensor mm(torch::Tensor A, torch::Tensor C, int64_t v) {
  auto out = torch::empty_like(A); int M = A.size(0);
  k<<<(M+127)/128,128>>>(A.data_ptr<double>(), C.data_ptr<double>(), out.data_ptr<double>(), M, (int)v); return out; }
"""
ext = load_inline(name="nl9_probe_mm", cpp_sources="torch::Tensor mm(torch::Tensor A, torch::Tensor C, int64_t v);",
                  cuda_sources=src, functions=["mm"], extra_cuda_cflags=["-O3", "--fmad=false", "-gencode=arch=compute_90,code=sm_90"])
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
for trial in range(3):
    C = (torch.rand((3, 3), generator=g, device=dev, dtype=torch.float64) - 0.3) * 30
    if trial == 0: C = torch.tril(C)
    for M in [1, 3, 27, 37, 648, 5184, 32864, 300000]:
        S = torch.randint(-150, 150, (M, 3), generator=g, device=dev).double()
        ref = S @ C
        X = torch.rand((M, 3), generator=g, device=dev, dtype=torch.float64) * 100
        refx = X @ C
        print(trial, M, [int((ext.mm(S, C, v) != ref).sum().item()) for v in range(4)],
              "float A:", [int((ext.mm(X, C, v) != refx).sum().item()) for v in range(4)], flush=True)
