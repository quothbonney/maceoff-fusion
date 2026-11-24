| system | path | step ms (synced, no profiler) | GPU kernel sum us | NL_build | conv_bwd(node) | gemm(radialMLP+linears) | elementwise | fused_node | conv_fwd(node) | cueq_SC | csr_sort/search | scatter/gather/index | fused_embed | scan/compact | reduce | memcpy/memset | other | host sync API calls/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wbox5184 | node-hybrid + proto | 66.176 | 65760 | 41995 | 6441 | 5981 | 5483 | 2370 | 2294 | 466 | 182 | 231 | 232 | 0 | 20 | 61 | 5 | {'cudaMemcpyAsync': 37.0, 'cudaStreamSynchronize': 29.0} |
| wbox5184 | GPUStep | 23.976 | 23864 | 52 | 6518 | 5975 | 5546 | 2367 | 2269 | 487 | 85 | 261 | 232 | 3 | 20 | 28 | 22 | {'cudaMemcpyAsync': 1.0} |
| water5184 | node-hybrid + proto | 22.673 | 22473 | 1247 | 5585 | 5233 | 4875 | 2365 | 2032 | 468 | 176 | 207 | 204 | 0 | 19 | 58 | 5 | {'cudaMemcpyAsync': 7.0, 'cudaStreamSynchronize': 1.0} |
| water5184 | GPUStep | 21.416 | 21320 | 49 | 5651 | 5229 | 4928 | 2365 | 2020 | 486 | 83 | 237 | 202 | 3 | 19 | 29 | 20 | {'cudaMemcpyAsync': 1.0} |
| wbox2187 | node-hybrid + proto | 18.619 | 17958 | 7331 | 2851 | 2636 | 2439 | 1021 | 1012 | 220 | 142 | 140 | 106 | 0 | 15 | 41 | 6 | {'cudaMemcpyAsync': 14.0, 'cudaStreamSynchronize': 7.0} |
| wbox2187 | GPUStep | 10.778 | 10685 | 40 | 2901 | 2639 | 2484 | 1022 | 1005 | 215 | 55 | 167 | 106 | 3 | 15 | 19 | 14 | {'cudaMemcpyAsync': 1.0} |
| water2187 | node-hybrid + proto | 9.672 | 9438 | 286 | 2340 | 2207 | 2109 | 1022 | 844 | 222 | 134 | 128 | 86 | 0 | 14 | 41 | 5 | {'cudaMemcpyAsync': 7.0, 'cudaStreamSynchronize': 1.0} |
| water2187 | GPUStep | 9.294 | 9185 | 35 | 2374 | 2204 | 2146 | 1023 | 839 | 219 | 54 | 153 | 89 | 3 | 14 | 19 | 13 | {'cudaMemcpyAsync': 1.0} |
| wbox648 | node-hybrid + proto | 5.007 | 4305 | 722 | 925 | 884 | 721 | 314 | 341 | 81 | 124 | 103 | 37 | 0 | 14 | 34 | 3 | {'cudaMemcpyAsync': 13.0, 'cudaStreamSynchronize': 3.0} |
| wbox648 | GPUStep | 3.744 | 3617 | 41 | 939 | 881 | 769 | 314 | 341 | 79 | 45 | 126 | 37 | 3 | 14 | 18 | 8 | {'cudaMemcpyAsync': 1.0} |
| aspirin | node-hybrid + proto | 1.048 | 746 | 22 | 93 | 110 | 227 | 109 | 21 | 25 | 49 | 46 | 18 | 0 | 11 | 15 | 1 | {'cudaMemcpyAsync': 7.0, 'cudaStreamSynchronize': 1.0} |
| aspirin | GPUStep | 0.880 | 756 | 15 | 99 | 108 | 267 | 105 | 21 | 24 | 9 | 61 | 18 | 3 | 10 | 9 | 7 | {'cudaMemcpyAsync': 1.0} |
