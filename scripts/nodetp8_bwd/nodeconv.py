"""NodeConvFunction (autograd) + install/uninstall for MACE 0.3.17 interaction blocks (phase8, agent C).

Forward: agent B's scripts/nodetp8_fwd/nodetp_fwd.py fwd() if results/nodetp8_fwd/READY exists (and import works),
else plain-torch reference_fwd. Backward: nodetp_bwd.bwd (fused src-centric kernel). First derivatives only
(the backward is not itself differentiable: create_graph=True / training forces -> error on double backward).

install(model, layers=None, fwd="auto"|"ref"|"B", csr_cache=True) -> handle; uninstall(model, handle=None) -> bool.
Mirrors flashtp_adapter_v2: class allowlist, block.conv_fusion = True, conv_tp replaced by NodeConv(orig) which keeps the
original module as `orig` (state_dict hooks expose stock keys); uninstall restores the exact original objects.
CSR: the first installed layer (producer) builds it on every call; with csr_cache=True later layers reuse it if they get
the very same edge_index tensor object at the same _version, N and capture state (MACE passes data["edge_index"] to
both layers), so one build per model call (also inside CUDA-graph capture; never reused across calls). Cost of build_csr: 2 stable sorts + 2 searchsorted on E (measured in READY).
"""
import os
import sys
import inspect
import weakref
import torch
from e3nn import o3

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from csr import build_csr  # noqa: E402
from layer_cfg import make_cfg, reference_fwd  # noqa: E402
import nodetp_bwd  # noqa: E402

R = os.environ.get("R", "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study")
SUPPORTED_INTERACTIONS = (
    "RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock", "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock", "RealAgnosticAttResidualInteractionBlock",
    "RealAgnosticResidualNonLinearInteractionBlock",
)
_STATE_ATTR = "_nodetp8_saved"


class NodeConvError(RuntimeError):
    pass


def _load_fwd(mode):
    if mode == "ref":
        return reference_fwd, "reference_fwd"
    ready = os.path.exists(f"{R}/results/nodetp8_fwd/READY")
    if mode == "B" or (mode == "auto" and ready):
        p = f"{R}/scripts/nodetp8_fwd"
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            import nodetp_fwd
            return nodetp_fwd.fwd, f"nodetp8_fwd.fwd ({nodetp_fwd.__file__})"
        except Exception as ex:  # noqa: BLE001
            if mode == "B":
                raise
            return reference_fwd, f"reference_fwd (B import failed: {ex!r})"
    return reference_fwd, "reference_fwd"


class NodeConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w, edge_index, csr_tuple, cfg, fwd_fn):
        csr = dict(zip(("perm_dst", "rowptr_dst", "perm_src", "rowptr_src"), csr_tuple))
        out = fwd_fn(cfg, x, y, w, edge_index, csr)
        ctx.save_for_backward(x, y, w, edge_index, *csr_tuple)
        ctx.cfg = cfg
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, y, w, edge_index, *ct = ctx.saved_tensors
        csr = dict(zip(("perm_dst", "rowptr_dst", "perm_src", "rowptr_src"), ct))
        gx, gy, gw = nodetp_bwd.bwd(ctx.cfg, grad_out, x, y, w, edge_index, csr)
        ni = ctx.needs_input_grad
        return (gx if ni[0] else None, gy if ni[1] else None, gw if ni[2] else None, None, None, None, None)


class _CsrCache:
    def __init__(self):
        self.ref = None
        self.ver = None
        self.N = None
        self.val = None
        self.cap = None

    def get(self, edge_index, N, enabled, producer):
        # producer (first installed layer) ALWAYS builds -> no reuse across model calls / across graph capture boundaries;
        # later layers reuse the producer's CSR only for the very same edge_index object/version/N and capture state.
        cap = torch.cuda.is_current_stream_capturing()
        if enabled and not producer and self.ref is not None and self.cap == cap and self.ref() is edge_index and self.ver == edge_index._version \
                and self.N == N:
            return self.val
        c = build_csr(edge_index, N)
        val = (c["perm_dst"], c["rowptr_dst"], c["perm_src"], c["rowptr_src"])
        if enabled:
            self.ref, self.ver, self.N, self.val, self.cap = weakref.ref(edge_index), edge_index._version, N, val, cap
        return val


class NodeConv(torch.nn.Module):
    def __init__(self, conv_tp, fwd_fn, fwd_name, cache, csr_cache=True, producer=True):
        super().__init__()
        self.orig = conv_tp
        self.cfg = make_cfg(conv_tp)
        self.cfg.conv_tp = conv_tp  # lets agent B fwd() reuse its own cfg cached on the module
        self.fwd_fn, self.fwd_name = fwd_fn, fwd_name
        self._cache = cache
        self.csr_cache = csr_cache
        self.producer = producer
        self.record = None  # optional list: append (x, y, w, edge_index) detached (testing)
        self._register_state_dict_hook(NodeConv._sd_hook)
        self._register_load_state_dict_pre_hook(NodeConv._load_hook, with_module=True)

    @staticmethod
    def _sd_hook(module, state_dict, prefix, local_metadata):
        inner = prefix + "orig."
        for k in [k for k in state_dict if k.startswith(inner)]:
            state_dict[prefix + k[len(inner):]] = state_dict.pop(k)
        md = getattr(state_dict, "_metadata", None)
        if md is not None:
            p = prefix[:-1] if prefix.endswith(".") else prefix
            for k in [k for k in md if k == p + ".orig" or k.startswith(p + ".orig.")]:
                md[p + k[len(p + ".orig"):]] = md.pop(k)
        return state_dict

    @staticmethod
    def _load_hook(module, state_dict, prefix, *args):
        for k in [k for k in state_dict if k.startswith(prefix) and not k.startswith(prefix + "orig.")]:
            state_dict[prefix + "orig." + k[len(prefix):]] = state_dict.pop(k)

    def __getstate__(self):
        raise NodeConvError("NodeConv is not picklable/deepcopyable: uninstall(model) first")

    def __deepcopy__(self, memo):
        self.__getstate__()

    def extra_repr(self):
        return f"node-centric fp64 conv d1={self.cfg.d1} d_out={self.cfg.d_out} W={self.cfg.W} fwd={self.fwd_name}"

    def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
        cfg = self.cfg
        n, e = node_feats.shape[0], edge_index.shape[1]
        if node_feats.shape != (n, cfg.d1) or edge_attrs.shape != (e, cfg.d2) or tp_weights.shape != (e, cfg.W):
            raise NodeConvError(f"shapes {tuple(node_feats.shape)} {tuple(edge_attrs.shape)} {tuple(tp_weights.shape)}")
        if node_feats.dtype != torch.float64 or not node_feats.is_cuda:
            raise NodeConvError(f"need cuda fp64, got {node_feats.dtype} {node_feats.device}")
        if self.record is not None:
            self.record.append(tuple(t.detach().clone() for t in (node_feats, edge_attrs, tp_weights, edge_index)))
        csr = self._cache.get(edge_index, n, self.csr_cache, self.producer)
        return NodeConvFunction.apply(node_feats.contiguous(), edge_attrs.contiguous(), tp_weights.contiguous(),
                                      edge_index, csr, cfg, self.fwd_fn)


def install(model, layers=None, fwd="auto", csr_cache=True):
    from mace.modules import blocks as mb
    if getattr(model, _STATE_ATTR, None):
        raise NodeConvError("nodeconv already installed")
    fwd_fn, fwd_name = _load_fwd(fwd)
    cache = _CsrCache()
    new = {}
    for li, block in enumerate(model.interactions):
        if layers is not None and li not in layers:
            continue
        cls = type(block)
        if cls.__name__ not in SUPPORTED_INTERACTIONS or cls is not getattr(mb, cls.__name__, None):
            raise NodeConvError(f"interactions.{li}: unsupported class {cls}")
        if 'hasattr(self, "conv_fusion")' not in inspect.getsource(cls.forward):
            raise NodeConvError(f"interactions.{li}: no conv_fusion branch")
        if hasattr(block, "conv_fusion") or type(block.conv_tp) is not o3.TensorProduct:
            raise NodeConvError(f"interactions.{li}: already fused / conv_tp is {type(block.conv_tp)}")
        new[li] = NodeConv(block.conv_tp, fwd_fn, fwd_name, cache, csr_cache, producer=not new)
    saved = []
    for li, nc in new.items():
        block = model.interactions[li]
        saved.append((li, block.conv_tp))
        block.conv_tp = nc
        block.conv_fusion = True
    object.__setattr__(model, _STATE_ATTR, saved)
    return saved


def uninstall(model, handle=None):
    saved = getattr(model, _STATE_ATTR, None) if handle is None else handle
    if not saved:
        return False
    for li, orig in saved:
        block = model.interactions[li]
        if isinstance(block.conv_tp, NodeConv):
            if block.conv_tp.orig is not orig:
                raise NodeConvError(f"interactions.{li}: NodeConv does not wrap saved conv_tp")
            block.conv_tp = orig
        block.__dict__.pop("conv_fusion", None)
    if hasattr(model, _STATE_ATTR):
        object.__delattr__(model, _STATE_ATTR)
    return True


def node_convs(model):
    return [b.conv_tp for b in model.interactions if isinstance(b.conv_tp, NodeConv)]
