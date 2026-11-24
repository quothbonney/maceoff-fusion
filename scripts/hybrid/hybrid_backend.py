"""Hybrid MACE backend (fp64): FlashTP (fp64-CG build, exp/fp64-cg) for conv_tp+scatter of both interaction layers,
cuEquivariance SymmetricContraction for both product blocks; everything else stays e3nn.

Design (option b): start from the unmodified e3nn model (enable_cueq=False). Everything in the e3nn model is in e3nn
"mul_ir" layout. (1) Install the FlashTP adapter (scripts/flash_proto/flashtp_adapter.py) -> conv_fusion path,
FlashConv returns mul_ir [N, irreps_out.dim]. (2) Convert a *separate* copy of the model with
mace.cli.convert_e3nn_cueq.run(layout="mul_ir", conv_fusion=False) and take products[i].symmetric_contractions
(cuet.SymmetricContraction, layout_in=ir_mul, layout_out=mul_ir, weights transferred by MACE's own converter).
CueqSC wraps it to the e3nn SC call signature: input [N, mul, d] (output of interaction.reshape, mul_ir) is transposed
to [N, d, mul] and flattened == ir_mul (all irreps in node_feats share mul=128), element index = argmax(node_attrs);
output is already mul_ir, so product.linear (e3nn) consumes it unchanged. This is exactly what MACE's own
EquivariantProductBasisBlock does when cueq_config.layout_str == "mul_ir".

Usage (venv-cueq; see results/hybrid/READY):  from hybrid_backend import build_calc, describe
"""
import os, sys, copy
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
_FP64CG = f"{R}/src/flashTP-fp64cg"
if _FP64CG not in sys.path:
    sys.path.insert(0, _FP64CG)
import flashTP_e3nn  # noqa: E402
assert flashTP_e3nn.__file__.startswith(_FP64CG), f"wrong flashTP_e3nn: {flashTP_e3nn.__file__}"
_FP = f"{R}/scripts/flash_proto"
if _FP not in sys.path:
    sys.path.insert(1, _FP)
import torch  # noqa: E402
import flashtp_adapter as FA  # noqa: E402
assert FA.flashTP_e3nn is flashTP_e3nn
from mace.calculators import MACECalculator  # noqa: E402


class CueqSC(torch.nn.Module):
    """e3nn-signature wrapper around cuet.SymmetricContraction (layout_in ir_mul, layout_out mul_ir)."""
    def __init__(self, cuet_sc, e3nn_sc):
        super().__init__()
        self.sc = cuet_sc
        self.irreps_in = e3nn_sc.irreps_in
        self.irreps_out = e3nn_sc.irreps_out
        muls = {mul for mul, _ in self.irreps_in}
        assert len(muls) == 1, f"non-uniform mul {self.irreps_in}: transpose(1,2) is not ir_mul"

    def forward(self, node_feats, node_attrs):  # node_feats [N, mul, d] mul_ir; node_attrs one-hot [N, n_el]
        x = torch.transpose(node_feats, 1, 2).flatten(1)
        idx = node_attrs.argmax(dim=-1).int()
        return self.sc(x, idx)


def _cueq_scs(model):
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    prev = torch.get_default_dtype()
    try:
        cm = to_cueq(copy.deepcopy(model), device="cuda", return_model=True, layout="mul_ir", conv_fusion=False)
    finally:
        torch.set_default_dtype(prev)
    return [p.symmetric_contractions for p in cm.products]


def install_cueq_sc(model):
    saved = []
    for i, (p, csc) in enumerate(zip(model.products, _cueq_scs(model))):
        assert not isinstance(p.symmetric_contractions, CueqSC)
        assert getattr(p, "cueq_config", None) is None or not p.cueq_config.enabled
        orig = p.symmetric_contractions
        p.symmetric_contractions = CueqSC(csc, orig).to(next(model.parameters()).device)
        saved.append((i, orig))
    return saved


def uninstall_cueq_sc(model, saved):
    for i, orig in saved:
        model.products[i].symmetric_contractions = orig


def _impl(mod):
    if isinstance(mod, FA.FlashConv):
        return f"FlashTP fp64-CG ({type(mod.ftp).__name__}) fused gather+TP+scatter"
    if isinstance(mod, CueqSC):
        return f"cuEquivariance {type(mod.sc).__module__}.{type(mod.sc).__name__}"
    return f"e3nn/torch {type(mod).__module__}.{type(mod).__name__}"


def describe(model):
    rows = {"node_embedding": _impl(model.node_embedding), "radial_embedding": _impl(model.radial_embedding),
            "spherical_harmonics": _impl(model.spherical_harmonics)}
    for i, b in enumerate(model.interactions):
        for n in ["linear_up", "conv_tp_weights", "conv_tp", "linear", "skip_tp"]:
            k = f"interactions.{i}.{n}" + ("(+scatter_sum)" if n == "conv_tp" else "")
            rows[k] = _impl(getattr(b, n))
        if not hasattr(b, "conv_fusion"):
            rows[f"interactions.{i}.scatter_sum"] = "torch scatter_sum"
    for i, p in enumerate(model.products):
        rows[f"products.{i}.symmetric_contractions"] = _impl(p.symmetric_contractions)
        rows[f"products.{i}.linear"] = _impl(p.linear)
    for i, r in enumerate(model.readouts):
        rows[f"readouts.{i}"] = _impl(r)
    return rows


def flash_cg_dtypes(model):
    """For each FlashConv: module dtype, dtypes of float tensors held, and whether the CG table is genuinely fp64
    (some CG values differ from their fp32-rounded value; the shipped fp32-rounded path would give False)."""
    out = {}
    for i, b in enumerate(model.interactions):
        if not isinstance(b.conv_tp, FA.FlashConv):
            continue
        dts, cgs = set(), []
        for m in b.conv_tp.ftp.modules():
            for k, v in vars(m).items():
                if torch.is_tensor(v) and v.is_floating_point():
                    dts.add(str(v.dtype))
                if k == "unique_cg_val" and v is not None:
                    cgs += [float(x) for x in (v.tolist() if torch.is_tensor(v) else v)]
            for v in m.buffers(recurse=False):
                if v.is_floating_point():
                    dts.add(str(v.dtype))
        import numpy as np
        out[f"interactions.{i}"] = dict(module_dtype=str(getattr(b.conv_tp.ftp, "dtype", None)), tensor_dtypes=sorted(dts),
                                        n_unique_cg=len(cgs),
                                        cg_not_fp32_representable=sum(float(np.float32(c)) != c for c in cgs))
    return out


def build_calc(model_path=f"{R}/models/MACE-OFF23_medium.model", device="cuda", conv="flashtp", sc="cueq"):
    """Return (MACECalculator, info). models[0] is modified in place: conv in {'flashtp','e3nn'}, sc in {'cueq','e3nn'}.
    Default = hybrid."""
    assert device == "cuda"
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64", compute_stress=False,
                          enable_cueq=False)
    model = calc.models[0]
    assert next(model.parameters()).dtype == torch.float64
    if sc == "cueq":
        install_cueq_sc(model)
    if conv == "flashtp":
        FA.install(model, dtype=torch.float64)  # FlashConv builds uvu_TP under set_default_dtype(float64)
    info = dict(variant=f"conv={conv},sc={sc}", flashTP_e3nn=flashTP_e3nn.__file__, flash_cg_dtypes=flash_cg_dtypes(model),
                backend_map=describe(model))
    if conv == "flashtp":
        for v in info["flash_cg_dtypes"].values():
            assert "torch.float32" not in v["tensor_dtypes"] and v["cg_not_fp32_representable"] > 0, info["flash_cg_dtypes"]
    return calc, info
