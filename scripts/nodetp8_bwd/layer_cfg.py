"""make_cfg(conv_tp) -> LayerCfg: everything needed to evaluate MACE's uvu conv_tp (e3nn o3.TensorProduct) exactly.

Semantics (e3nn 0.4.4, component irrep normalisation, element path normalisation; path_weight taken from the module):
  for path p = (i1, i2, io; l1, l2, l3), mul channel u, components i (x), j (y), k (out):
    tp[e, out_off[p] + u*(2l3+1) + k] += path_weight[p] * sum_{i,j} C_p[i,j,k] * x[e, x_off[p] + u*(2l1+1) + i]
                                                           * y[e, y_off[p] + j] * w[e, w_off[p] + u]
  with C_p = o3.wigner_3j(l1, l2, l3, dtype=float64) (full fp64), w_off[p] = p * mul (instruction-major chunks).

Fields (plain python; tensors are CPU fp64/int unless noted):
  d1, d2, d_out, W, mul (all irreps share one multiplicity; asserted)
  x_irreps / y_irreps / out_irreps: list of (mul, l, parity, offset, dim_per_mul)
  paths: list of dict(i1,i2,io,l1,l2,l3,x_off,y_off,out_off,w_off,path_weight,cg (dense fp64 [2l1+1,2l2+1,2l3+1]),
                      nz: list of (i, j, k, value) nonzero CG entries in row-major order (fp64 python floats))
  packed: cg_ijk int32 [nnz,3], cg_val fp64 [nnz], cg_path_ptr int32 [P+1]
"""
from dataclasses import dataclass, field
import torch
from e3nn import o3


@dataclass
class LayerCfg:
    d1: int
    d2: int
    d_out: int
    W: int
    mul: int
    x_irreps: list
    y_irreps: list
    out_irreps: list
    paths: list
    cg_ijk: torch.Tensor = None
    cg_val: torch.Tensor = None
    cg_path_ptr: torch.Tensor = None
    key: str = ""
    _dev_cache: dict = field(default_factory=dict, repr=False)

    def dense_cg(self, p, device):
        k = ("cg", p, str(device))
        if k not in self._dev_cache:
            self._dev_cache[k] = self.paths[p]["cg"].to(device)
        return self._dev_cache[k]


def _irreps_table(irreps):
    out, off = [], 0
    for mul, ir in irreps:
        out.append((mul, ir.l, ir.p, off, ir.dim))
        off += mul * ir.dim
    return out


def make_cfg(conv_tp):
    if type(conv_tp) is not o3.TensorProduct:
        raise TypeError(f"expected e3nn o3.TensorProduct, got {type(conv_tp)}")
    if conv_tp.internal_weights or conv_tp.shared_weights:
        raise ValueError("conv_tp must use external per-edge weights")
    xi, yi, oi = (_irreps_table(conv_tp.irreps_in1), _irreps_table(conv_tp.irreps_in2), _irreps_table(conv_tp.irreps_out))
    muls = {m for m, *_ in xi} | {m for m, *_ in oi}
    if len(muls) != 1 or any(m != 1 for m, *_ in yi):
        raise ValueError(f"need uniform mul in x/out and mul 1 in y: {conv_tp.irreps_in1} {conv_tp.irreps_in2} {conv_tp.irreps_out}")
    mul = muls.pop()
    paths, w_off = [], 0
    for ins in conv_tp.instructions:
        if ins.connection_mode != "uvu" or not ins.has_weight:
            raise ValueError(f"unsupported instruction {ins}")
        l1, l2, l3 = xi[ins.i_in1][1], yi[ins.i_in2][1], oi[ins.i_out][1]
        cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu").contiguous()
        nz = [(i, j, k, float(cg[i, j, k])) for i in range(2 * l1 + 1) for j in range(2 * l2 + 1)
              for k in range(2 * l3 + 1) if float(cg[i, j, k]) != 0.0]
        paths.append(dict(i1=ins.i_in1, i2=ins.i_in2, io=ins.i_out, l1=l1, l2=l2, l3=l3, x_off=xi[ins.i_in1][3],
                          y_off=yi[ins.i_in2][3], out_off=oi[ins.i_out][3], w_off=w_off,
                          path_weight=float(ins.path_weight), cg=cg, nz=nz))
        assert tuple(ins.path_shape) == (mul, 1), ins.path_shape
        w_off += mul
    assert w_off == conv_tp.weight_numel
    ijk, val, ptr = [], [], [0]
    for p in paths:
        for (i, j, k, v) in p["nz"]:
            ijk.append((i, j, k))
            val.append(v)
        ptr.append(len(val))
    key = repr([(p["l1"], p["l2"], p["l3"], p["x_off"], p["y_off"], p["out_off"], p["w_off"], p["path_weight"])
                for p in paths]) + f"|{conv_tp.irreps_in1.dim},{conv_tp.irreps_in2.dim},{conv_tp.irreps_out.dim},{mul}"
    return LayerCfg(d1=conv_tp.irreps_in1.dim, d2=conv_tp.irreps_in2.dim, d_out=conv_tp.irreps_out.dim,
                    W=conv_tp.weight_numel, mul=mul, x_irreps=xi, y_irreps=yi, out_irreps=oi, paths=paths,
                    cg_ijk=torch.tensor(ijk, dtype=torch.int32), cg_val=torch.tensor(val, dtype=torch.float64),
                    cg_path_ptr=torch.tensor(ptr, dtype=torch.int32), key=key)


# ------------------------------------------------------------------------------------------ plain-torch references
def tp_edges(cfg, xe, y, w):
    """Per-edge TP in plain torch: xe [E,d1] (already gathered), y [E,d2], w [E,W] -> [E,d_out]."""
    E = xe.shape[0]
    out = xe.new_zeros((E, cfg.d_out))
    m = cfg.mul
    for p, P in enumerate(cfg.paths):
        a, b, c = 2 * P["l1"] + 1, 2 * P["l2"] + 1, 2 * P["l3"] + 1
        xb = xe[:, P["x_off"]:P["x_off"] + m * a].reshape(E, m, a)
        yb = y[:, P["y_off"]:P["y_off"] + b]
        wb = w[:, P["w_off"]:P["w_off"] + m]
        cg = cfg.dense_cg(p, xe.device)
        t = torch.einsum("eui,ej,ijk->euk", xb, yb, cg) * wb[:, :, None]
        out[:, P["out_off"]:P["out_off"] + m * c] += P["path_weight"] * t.reshape(E, m * c)
    return out


def reference_fwd(cfg, x, y, w, edge_index, csr=None):
    N = x.shape[0]
    msg = tp_edges(cfg, x[edge_index[0]], y, w)
    out = x.new_zeros((N, cfg.d_out))
    out.index_add_(0, edge_index[1], msg)
    return out


def reference_bwd(cfg, grad_out, x, y, w, edge_index, csr=None):
    """Analytic first-derivative backward in plain torch -> (gx [N,d1], gy [E,d2], gw [E,W])."""
    N, E, m = x.shape[0], y.shape[0], cfg.mul
    xe = x[edge_index[0]]
    ge = grad_out[edge_index[1]]
    gxe = x.new_zeros((E, cfg.d1))
    gy = x.new_zeros((E, cfg.d2))
    gw = x.new_zeros((E, cfg.W))
    for p, P in enumerate(cfg.paths):
        a, b, c = 2 * P["l1"] + 1, 2 * P["l2"] + 1, 2 * P["l3"] + 1
        xb = xe[:, P["x_off"]:P["x_off"] + m * a].reshape(E, m, a)
        yb = y[:, P["y_off"]:P["y_off"] + b]
        wb = w[:, P["w_off"]:P["w_off"] + m]
        gb = ge[:, P["out_off"]:P["out_off"] + m * c].reshape(E, m, c) * P["path_weight"]
        cg = cfg.dense_cg(p, x.device)
        gw[:, P["w_off"]:P["w_off"] + m] += torch.einsum("eui,ej,ijk,euk->eu", xb, yb, cg, gb)
        gy[:, P["y_off"]:P["y_off"] + b] += torch.einsum("eui,eu,ijk,euk->ej", xb, wb, cg, gb)
        gxe[:, P["x_off"]:P["x_off"] + m * a] += torch.einsum("ej,eu,ijk,euk->eui", yb, wb, cg, gb).reshape(E, m * a)
    gx = x.new_zeros((N, cfg.d1))
    gx.index_add_(0, edge_index[0], gxe)
    return gx, gy, gw
