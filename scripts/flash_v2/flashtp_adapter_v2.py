"""FlashTP adapter v2 for MACE 0.3.17 interaction blocks (experimental, hardened).

Replacement boundary (per interaction block), unchanged from v1:
    mji = conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights); message = scatter_sum(mji, edge_index[1], N)
is replaced by MACE's own `if hasattr(self, "conv_fusion")` branch, which calls
    message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
so we swap conv_tp for FlashConv and set block.conv_fusion = True. Nothing on disk is modified.

v2 changes vs scripts/flash_proto/flashtp_adapter.py (v1):
  * FlashTP is imported from the cgpatch worktree ($R/src/flashTP-cgpatch, commit 0e4d820): the CG table precision follows
    the `dtype` argument, so we no longer touch torch's global default dtype while building. The build asserts the CG table
    is fp64 and actually holds values that are not fp32-representable, and that every nonzero e3nn fp64 wigner_3j entry of
    every path is within 1e-14 of a table value.
  * Zero-edge graphs (or zero nodes) never reach the kernel (the kernel fails with "invalid configuration argument" at
    E=0): FlashConv returns zeros [N, irreps_out.dim] through an autograd.Function whose backward returns zeros of the
    input shapes (same values/shapes as e3nn's gather+TP+scatter_sum for E=0).
  * Validation: interaction class allowlist (exact class, mace 0.3.17, forward has the conv_fusion branch), conv_tp must be
    a plain e3nn o3.TensorProduct with uvu+weighted instructions, external weights, e3nn-default path weights; rejects
    Magnetic*, cueq/oeq-configured, already-fused blocks. Per call: input dtype/device/shape match the built TP,
    n_nodes/n_edges < 2^31, optional index range check (default on; one aminmax + host sync per layer).
  * install() twice -> AdapterError; uninstall() is idempotent and restores the original module objects.
  * state_dict while installed yields exactly the stock MACE keys (FlashConv keeps the original conv_tp as submodule
    `orig`; a state_dict hook strips the `orig.` prefix, a load pre-hook adds it). Pickling/deepcopy of a FlashConv
    (torch.save(model), copy.deepcopy(model)) raises: uninstall first (see LIMITATIONS.md).
  * int32 edge index conversion every call (no caching), as in v1.
"""
import os
import sys
import inspect

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
CGPATCH = f"{R}/src/flashTP-cgpatch"
if "flashTP_e3nn" in sys.modules and not sys.modules["flashTP_e3nn"].__file__.startswith(CGPATCH + "/"):
    raise ImportError(f"flashTP_e3nn already imported from {sys.modules['flashTP_e3nn'].__file__}; "
                      f"v2 needs {CGPATCH} (import flashtp_adapter_v2 before anything else imports flashTP_e3nn)")
if CGPATCH not in sys.path:
    sys.path.insert(0, CGPATCH)
import flashTP_e3nn  # noqa: E402
assert flashTP_e3nn.__file__.startswith(CGPATCH + "/"), f"wrong flashTP_e3nn: {flashTP_e3nn.__file__}"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from e3nn import o3  # noqa: E402
import mace  # noqa: E402
from mace.modules import blocks as mace_blocks  # noqa: E402

TESTED_MACE = "0.3.17"
SUPPORTED_INTERACTIONS = (
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock",
    "RealAgnosticAttResidualInteractionBlock",
    "RealAgnosticResidualNonLinearInteractionBlock",
)
INT32_MAX = 2**31 - 1
_STATE_ATTR = "_flashtp_v2_saved"


class AdapterError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------- checks
def _check_block(li, block):
    cls = type(block)
    name = cls.__name__
    if name.startswith("Magnetic") or "magmom_conv_tp" in block._modules:
        raise AdapterError(f"interactions.{li}: {name} is a magnetic interaction (no conv_fusion branch); unsupported")
    if name not in SUPPORTED_INTERACTIONS or cls is not getattr(mace_blocks, name, None):
        raise AdapterError(f"interactions.{li}: unsupported interaction class {cls.__module__}.{name}; "
                           f"supported (exact class, mace {TESTED_MACE}): {SUPPORTED_INTERACTIONS}")
    if 'hasattr(self, "conv_fusion")' not in inspect.getsource(cls.forward):
        raise AdapterError(f"interactions.{li}: {name}.forward has no conv_fusion branch in this MACE build")
    if hasattr(block, "conv_fusion") or isinstance(block.conv_tp, FlashConv):
        raise AdapterError(f"interactions.{li}: block already fused (conv_fusion set: FlashTP, cueq or oeq conv fusion)")
    for cfg in ("cueq_config", "oeq_config"):
        c = getattr(block, cfg, None)
        if c is not None and getattr(c, "enabled", False):
            raise AdapterError(f"interactions.{li}: {cfg}.enabled=True (cueq/oeq model); convert to e3nn first")
    tp = block.conv_tp
    if type(tp) is not o3.TensorProduct:
        raise AdapterError(f"interactions.{li}: conv_tp is {type(tp).__module__}.{type(tp).__name__}, "
                           "expected plain e3nn o3.TensorProduct")
    bad = [(i.connection_mode, i.has_weight) for i in tp.instructions if i.connection_mode != "uvu" or not i.has_weight]
    if bad:
        raise AdapterError(f"interactions.{li}: conv_tp has non-uvu or unweighted instructions {bad}")
    if tp.internal_weights or tp.shared_weights:
        raise AdapterError(f"interactions.{li}: conv_tp must use external per-edge weights")
    # FlashTP rebuilds instructions with e3nn defaults (component/element normalisation): path weights must agree.
    ref = o3.TensorProduct(tp.irreps_in1, tp.irreps_in2, tp.irreps_out,
                           [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions],
                           shared_weights=False, internal_weights=False)
    pw = [(a.path_weight, b.path_weight) for a, b in zip(tp.instructions, ref.instructions)]
    if any(abs(a - b) > 1e-15 * max(1.0, abs(b)) for a, b in pw):
        raise AdapterError(f"interactions.{li}: conv_tp path weights differ from e3nn defaults {pw}")
    for (mul, _ir) in tp.irreps_in1:
        if mul % 32:
            raise AdapterError(f"interactions.{li}: irreps_in1 multiplicity {mul} not a multiple of 32 (FlashTP)")


def cg_report(ftp, conv_tp, dtype):
    """CG table of a built FlashTP module: dtype, size, #values not fp32-representable, max distance of e3nn fp64
    wigner_3j nonzeros to the table."""
    vals = [float(v) for v in ftp.unique_cg_val]
    tabs = [t for t in ftp.metadata_list if torch.is_tensor(t) and t.is_floating_point()]
    t_cg = [t for t in tabs if t.numel() == len(vals)]
    table = np.array(sorted(vals))
    max_dist = 0.0
    for ins in conv_tp.instructions:
        l1, l2, l3 = conv_tp.irreps_in1[ins.i_in1].ir.l, conv_tp.irreps_in2[ins.i_in2].ir.l, conv_tp.irreps_out[ins.i_out].ir.l
        w = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
        nz = w[w != 0].numpy()
        if nz.size:
            idx = np.clip(np.searchsorted(table, nz), 1, len(table) - 1)
            d = np.minimum(np.abs(table[idx] - nz), np.abs(table[idx - 1] - nz))
            max_dist = max(max_dist, float(d.max()))
    return dict(kernel_class=type(ftp).__name__, n_unique_cg=len(vals),
                cg_not_fp32_representable=int(sum(float(np.float32(v)) != v for v in vals)),
                float_tensor_dtypes=sorted({str(t.dtype) for t in tabs}),
                cg_tensor_dtype=str(t_cg[0].dtype) if t_cg else None,
                max_abs_e3nn_fp64_cg_to_table=max_dist, requested_dtype=str(dtype),
                default_dtype_at_build=str(torch.get_default_dtype()))


# ------------------------------------------------------------------------------------------------------ zero edges
class _ZeroConv(torch.autograd.Function):
    """E == 0 (or N == 0): message is zeros [N, out_dim]; gradients are zeros of the input shapes (as e3nn)."""

    @staticmethod
    def forward(ctx, node_feats, edge_attrs, tp_weights, out_dim):
        ctx.shapes = (node_feats.shape, edge_attrs.shape, tp_weights.shape)
        ctx.dd = (node_feats.dtype, node_feats.device)
        return node_feats.new_zeros((node_feats.shape[0], out_dim))

    @staticmethod
    def backward(ctx, grad):
        dt, dev = ctx.dd
        g = [torch.zeros(s, dtype=dt, device=dev) if need else None
             for s, need in zip(ctx.shapes, ctx.needs_input_grad[:3])]
        return g[0], g[1], g[2], None


# ------------------------------------------------------------------------------------------------------- FlashConv
class FlashConv(torch.nn.Module):
    """Fused gather + uvu TP + scatter_sum (FlashTP, cgpatch). Keeps the original e3nn conv_tp as `orig` so the module
    state_dict keys are the stock ones (hooks strip/add the `orig.` prefix); `orig` is never called."""

    def __init__(self, conv_tp, dtype, device, check_indices=True):
        super().__init__()
        self.orig = conv_tp
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise AdapterError(f"FlashTP needs a CUDA device, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if dtype not in (torch.float32, torch.float64):
            raise AdapterError(f"unsupported dtype {dtype}")
        insts = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in conv_tp.instructions]
        with torch.cuda.device(self.device):
            # NB: no torch.set_default_dtype here: the cgpatch build takes CG precision from `dtype`.
            self.ftp = flashTP_e3nn.uvu_TP(conv_tp.irreps_in1, conv_tp.irreps_in2, conv_tp.irreps_out, insts,
                                           device=self.device, dtype=dtype)
        self.in1_dim = conv_tp.irreps_in1.dim
        self.in2_dim = conv_tp.irreps_in2.dim
        self.out_dim = conv_tp.irreps_out.dim
        self.irreps_out = conv_tp.irreps_out
        self.weight_numel = conv_tp.weight_numel
        self.check_indices = check_indices
        self.cg = cg_report(self.ftp, conv_tp, dtype)
        if dtype == torch.float64:
            if not (self.cg["cg_tensor_dtype"] == "torch.float64" and self.cg["cg_not_fp32_representable"] > 0
                    and "torch.float32" not in self.cg["float_tensor_dtypes"]
                    and self.cg["max_abs_e3nn_fp64_cg_to_table"] < 1e-14):
                raise AdapterError(f"FlashTP CG table is not genuine fp64: {self.cg}")
        if any(True for _ in self.ftp.state_dict()):
            raise AdapterError("FlashTP module unexpectedly carries state_dict entries")
        self._register_state_dict_hook(FlashConv._sd_hook)
        self._register_load_state_dict_pre_hook(FlashConv._load_hook, with_module=True)

    # ---- state_dict: expose stock keys
    @staticmethod
    def _sd_hook(module, state_dict, prefix, local_metadata):
        inner = prefix + "orig."
        for k in [k for k in state_dict if k.startswith(prefix)]:
            v = state_dict.pop(k)
            if k.startswith(inner):
                state_dict[prefix + k[len(inner):]] = v
        md = getattr(state_dict, "_metadata", None)
        if md is not None:
            p = prefix[:-1] if prefix.endswith(".") else prefix
            for k in [k for k in md if k == p + ".orig" or k.startswith(p + ".orig.") or k.startswith(p + ".ftp")]:
                v = md.pop(k)
                if k.startswith(p + ".orig"):
                    nk = p + k[len(p + ".orig"):]
                    md[nk] = v
        return state_dict

    @staticmethod
    def _load_hook(module, state_dict, prefix, *args):
        for k in [k for k in state_dict if k.startswith(prefix) and not k.startswith(prefix + "orig.")]:
            state_dict[prefix + "orig." + k[len(prefix):]] = state_dict.pop(k)

    # ---- pickling guard
    def __getstate__(self):
        raise AdapterError("FlashConv (FlashTP v2 adapter) is not picklable/deepcopyable: call "
                           "flashtp_adapter_v2.uninstall(model) before torch.save(model)/copy.deepcopy(model); "
                           "model.state_dict() is safe while installed (stock keys)")

    def __deepcopy__(self, memo):
        self.__getstate__()

    def extra_repr(self):
        return f"FlashTP {self.cg['kernel_class']} dtype={self.dtype} device={self.device} check_indices={self.check_indices}"

    def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
        if node_feats.dim() != 2 or node_feats.shape[1] != self.in1_dim:
            raise AdapterError(f"node_feats shape {tuple(node_feats.shape)}, expected [N, {self.in1_dim}]")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise AdapterError(f"edge_index shape {tuple(edge_index.shape)}, expected [2, E]")
        if edge_index.dtype not in (torch.int64, torch.int32):
            raise AdapterError(f"edge_index dtype {edge_index.dtype}, expected int64/int32")
        n, e = node_feats.shape[0], edge_index.shape[1]
        if tuple(edge_attrs.shape) != (e, self.in2_dim):
            raise AdapterError(f"edge_attrs shape {tuple(edge_attrs.shape)}, expected [{e}, {self.in2_dim}]")
        if tuple(tp_weights.shape) != (e, self.weight_numel):
            raise AdapterError(f"tp_weights shape {tuple(tp_weights.shape)}, expected [{e}, {self.weight_numel}]")
        for name, t in (("node_feats", node_feats), ("edge_attrs", edge_attrs), ("tp_weights", tp_weights)):
            if t.dtype != self.dtype:
                raise AdapterError(f"{name} dtype {t.dtype} != FlashTP build dtype {self.dtype}")
            if t.device != self.device:
                raise AdapterError(f"{name} on {t.device} != FlashTP build device {self.device}")
        if edge_index.device != self.device:
            raise AdapterError(f"edge_index on {edge_index.device} != FlashTP build device {self.device}")
        if n > INT32_MAX or e > INT32_MAX:
            raise AdapterError(f"n_nodes={n} / n_edges={e} exceed int32 range")
        if e == 0 or n == 0:
            return _ZeroConv.apply(node_feats, edge_attrs, tp_weights, self.out_dim)
        if self.check_indices:
            mn, mx = torch.aminmax(edge_index)
            mn, mx = int(mn), int(mx)
            if mn < 0 or mx >= n:
                raise AdapterError(f"edge_index out of range [{mn}, {mx}] for n_nodes={n}")
        # int32 conversion every call (v0 cached by data_ptr -> stale graphs).
        src = edge_index[0].to(torch.int32).contiguous()
        dst = edge_index[1].to(torch.int32).contiguous()
        with torch.cuda.device(self.device):
            return self.ftp(node_feats.contiguous(), edge_attrs.contiguous(), tp_weights.contiguous(), src, dst)


# -------------------------------------------------------------------------------------------- install / uninstall
def check_model(model, layers=None):
    if mace.__version__ != TESTED_MACE:
        raise AdapterError(f"mace {mace.__version__} untested (adapter validated on {TESTED_MACE})")
    if is_installed(model):
        raise AdapterError("FlashTP v2 adapter already installed on this model; uninstall(model) first")
    for li, block in enumerate(model.interactions):
        if layers is None or li in layers:
            _check_block(li, block)


def install(model, dtype=None, device=None, layers=None, check_indices=True):
    """Replace conv_tp in the given interaction layers (default: all). All checks run before any module is modified.
    dtype/device default to the model's parameters. Returns the list of (layer, original conv_tp)."""
    check_model(model, layers)
    p = next(model.parameters())
    dtype = dtype or p.dtype
    device = torch.device(device) if device is not None else p.device
    if p.dtype != dtype:
        raise AdapterError(f"model parameters are {p.dtype}, requested FlashTP dtype {dtype}")
    if p.device.type != "cuda" or (device.index is not None and p.device != device):
        raise AdapterError(f"model parameters on {p.device}, requested FlashTP device {device}")
    new = {}
    for li, block in enumerate(model.interactions):
        if layers is None or li in layers:
            new[li] = FlashConv(block.conv_tp, dtype, device, check_indices=check_indices)  # may raise; nothing swapped yet
    saved = []
    for li, fc in new.items():
        block = model.interactions[li]
        saved.append((li, block.conv_tp))
        block.conv_tp = fc
        block.conv_fusion = True
    object.__setattr__(model, _STATE_ATTR, saved)  # plain attribute (not a module/buffer)
    return saved


def uninstall(model, saved=None):
    """Restore the original conv_tp modules and remove conv_fusion. Idempotent; returns True if something was undone."""
    saved = getattr(model, _STATE_ATTR, None) if saved is None else saved
    if not saved:
        return False
    did = False
    for li, orig in saved:
        block = model.interactions[li]
        if isinstance(block.conv_tp, FlashConv):
            if block.conv_tp.orig is not orig:
                raise AdapterError(f"interactions.{li}: FlashConv does not wrap the saved original conv_tp")
            block.conv_tp = orig
            did = True
        if "conv_fusion" in block.__dict__:
            del block.__dict__["conv_fusion"]
            did = True
    if hasattr(model, _STATE_ATTR):
        object.__delattr__(model, _STATE_ATTR)
    return did


def is_installed(model):
    return any(isinstance(b.conv_tp, FlashConv) for b in model.interactions)


def flash_cg_reports(model):
    return {f"interactions.{i}": b.conv_tp.cg for i, b in enumerate(model.interactions) if isinstance(b.conv_tp, FlashConv)}
