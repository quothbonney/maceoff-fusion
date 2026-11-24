"""Hybrid MACE backend v2: FlashTP (cgpatch, fp64 CG) conv_tp+scatter and/or cuEquivariance SymmetricContraction on top
of the unmodified e3nn model. Based on scripts/hybrid/hybrid_backend.py (v1); differences:
  * uses flashtp_adapter_v2 (cgpatch FlashTP, zero-edge handling, validation, stock state_dict keys);
  * build_calc takes `dtype` and never sets torch's global default dtype itself (the caller's default is restored after
    MACE's own converters, which set it internally); works under default float32 or float64;
  * CueqSC install/uninstall are guarded (double install -> error, uninstall idempotent). sc=cueq changes the
    products.*.symmetric_contractions state_dict keys (cuet layout); see LIMITATIONS.md.

build_calc(model_path, device="cuda", conv="flashtp"|"e3nn", sc="cueq"|"e3nn", dtype=torch.float64) -> (calc, info)
"""
import os
import sys
import copy

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import flashtp_adapter_gc as FA  # noqa: E402  (inserts $R/src/flashTP-cgpatch and asserts flashTP_e3nn comes from it)
import torch  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

R = FA.R
flashTP_e3nn = FA.flashTP_e3nn
MODEL = f"{R}/models/MACE-OFF23_medium.model"
_SC_ATTR = "_cueq_sc_v2_saved"


class CueqSC(torch.nn.Module):
    """e3nn-signature wrapper around cuet.SymmetricContraction (layout_in ir_mul, layout_out mul_ir). Unchanged from v1."""

    def __init__(self, cuet_sc, e3nn_sc):
        super().__init__()
        self.sc = cuet_sc
        self.irreps_in = e3nn_sc.irreps_in
        self.irreps_out = e3nn_sc.irreps_out
        muls = {mul for mul, _ in self.irreps_in}
        if len(muls) != 1:
            raise FA.AdapterError(f"non-uniform mul {self.irreps_in}: transpose(1,2) is not ir_mul")

    def forward(self, node_feats, node_attrs):  # node_feats [N, mul, d] mul_ir; node_attrs one-hot [N, n_el]
        x = torch.transpose(node_feats, 1, 2).flatten(1)
        idx = node_attrs.argmax(dim=-1).int()
        return self.sc(x, idx)


def _cueq_scs(model, device):
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    prev = torch.get_default_dtype()
    try:  # MACE's converter sets the default dtype to the model dtype internally (and restores it); belt and braces
        cm = to_cueq(copy.deepcopy(model), device=str(device), return_model=True, layout="mul_ir", conv_fusion=False)
    finally:
        torch.set_default_dtype(prev)
    return [p.symmetric_contractions for p in cm.products]


def install_cueq_sc(model):
    if getattr(model, _SC_ATTR, None):
        raise FA.AdapterError("cueq SymmetricContraction already installed")
    if FA.is_installed(model):
        raise FA.AdapterError("install cueq SC before FlashTP (conversion deep-copies the model)")
    for i, p in enumerate(model.products):
        if isinstance(p.symmetric_contractions, CueqSC):
            raise FA.AdapterError(f"products.{i}: already CueqSC")
        c = getattr(p, "cueq_config", None)
        if c is not None and getattr(c, "enabled", False):
            raise FA.AdapterError(f"products.{i}: model already cueq-enabled")
    dev = next(model.parameters()).device
    saved = []
    for i, (p, csc) in enumerate(zip(model.products, _cueq_scs(model, dev))):
        orig = p.symmetric_contractions
        p.symmetric_contractions = CueqSC(csc, orig).to(dev)
        saved.append((i, orig))
    object.__setattr__(model, _SC_ATTR, saved)
    return saved


def uninstall_cueq_sc(model):
    saved = getattr(model, _SC_ATTR, None)
    if not saved:
        return False
    for i, orig in saved:
        model.products[i].symmetric_contractions = orig
    object.__delattr__(model, _SC_ATTR)
    return True


def _impl(mod):
    if isinstance(mod, FA.FlashConv):
        return f"FlashTP cgpatch ({mod.cg['kernel_class']}, {mod.dtype}) fused gather+TP+scatter"
    if isinstance(mod, CueqSC):
        return f"cuEquivariance {type(mod.sc).__module__}.{type(mod.sc).__name__}"
    return f"e3nn/torch {type(mod).__module__}.{type(mod).__name__}"


def describe(model):
    rows = {"node_embedding": _impl(model.node_embedding), "radial_embedding": _impl(model.radial_embedding),
            "spherical_harmonics": _impl(model.spherical_harmonics)}
    for i, b in enumerate(model.interactions):
        for n in ["linear_up", "conv_tp_weights", "conv_tp", "linear", "skip_tp"]:
            if hasattr(b, n):
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


def cueq_sc_dtypes(model):
    out = {}
    for i, p in enumerate(model.products):
        if isinstance(p.symmetric_contractions, CueqSC):
            sc = p.symmetric_contractions.sc
            out[f"products.{i}"] = dict(param_dtypes=sorted({str(t.dtype) for t in sc.parameters()}),
                                        math_dtype=str(getattr(sc, "math_dtype", None)))
    return out


def build_calc(model_path=MODEL, device="cuda", conv="flashtp", sc="cueq", dtype=torch.float64, check_indices=True):
    """Return (MACECalculator, info). calc.models[0] is modified in place.
    conv in {'flashtp','e3nn'}, sc in {'cueq','e3nn'}; default = hybrid. torch's global default dtype is left as the
    caller had it (MACECalculator/converters scope their own changes)."""
    if conv not in ("flashtp", "e3nn") or sc not in ("cueq", "e3nn"):
        raise ValueError(f"conv={conv!r} sc={sc!r}")
    if torch.device(device).type != "cuda":
        raise ValueError("cuda only")
    dname = {torch.float64: "float64", torch.float32: "float32"}[dtype]
    default_before = torch.get_default_dtype()
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype=dname, compute_stress=False,
                          enable_cueq=False)
    if torch.get_default_dtype() != default_before:  # MACE may change it; restore the caller's (do not impose ours)
        torch.set_default_dtype(default_before)
    model = calc.models[0]
    if next(model.parameters()).dtype != dtype:
        raise FA.AdapterError(f"model dtype {next(model.parameters()).dtype} != {dtype}")
    if sc == "cueq":
        install_cueq_sc(model)
    if conv == "flashtp":
        FA.install(model, dtype=dtype, check_indices=check_indices)
    if torch.get_default_dtype() != default_before:
        raise FA.AdapterError(f"global default dtype changed during build: {default_before} -> {torch.get_default_dtype()}")
    info = dict(variant=f"conv={conv},sc={sc}", dtype=str(dtype), default_dtype_at_build=str(default_before),
                flashTP_e3nn=flashTP_e3nn.__file__, flash_cg=FA.flash_cg_reports(model),
                cueq_sc=cueq_sc_dtypes(model), backend_map=describe(model))
    return calc, info
