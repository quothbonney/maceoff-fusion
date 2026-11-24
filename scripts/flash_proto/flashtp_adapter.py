"""Reversible, in-memory FlashTP adapter for MACE (v0.3.17) interaction blocks.

Replacement boundary (per interaction block): the e3nn edge tensor product + neighbour scatter_sum, i.e.
    mji = conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights); message = scatter_sum(mji, edge_index[1], N)
MACE's own `if hasattr(self, "conv_fusion")` branch (blocks.py:918-919 / ~1021) calls
    message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
so we swap conv_tp for FlashConv and set `conv_fusion`. Radial MLP, linear_up, linear, /avg_num_neighbors,
skip_tp, symmetric contraction are untouched. Nothing on disk is modified; `uninstall` restores the original modules.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # e3nn 0.4.4 under torch>=2.6 (mace sets the same)
import torch
import flashTP_e3nn


class FlashConv(torch.nn.Module):
    def __init__(self, conv_tp, dtype, device="cuda"):
        super().__init__()
        insts = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in conv_tp.instructions]
        assert all(m == "uvu" and w for (_, _, _, m, w) in insts), insts
        prev = torch.get_default_dtype(); torch.set_default_dtype(dtype)
        try:
            self.ftp = flashTP_e3nn.uvu_TP(conv_tp.irreps_in1, conv_tp.irreps_in2, conv_tp.irreps_out, insts,
                                           device=device, dtype=dtype)
        finally:
            torch.set_default_dtype(prev)
        self.weight_numel = conv_tp.weight_numel
        self.irreps_out = conv_tp.irreps_out
        # NOTE: an earlier version cached the int32 edge index keyed on (data_ptr, n_edges, _version); the CUDA caching
        # allocator reuses addresses across calls, so the cache served stale graphs (wrong forces on changing coordinates).
        # Convert every call (cost: two small int64->int32 casts).

    def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
        src = edge_index[0].to(torch.int32).contiguous()
        dst = edge_index[1].to(torch.int32).contiguous()
        return self.ftp(node_feats.contiguous(), edge_attrs.contiguous(), tp_weights.contiguous(), src, dst)


def install(model, dtype=None, layers=None):
    """Replace conv_tp in the given interaction layers (default: all). Returns a handle for uninstall()."""
    dtype = dtype or next(model.parameters()).dtype
    saved = []
    for li, block in enumerate(model.interactions):
        if layers is not None and li not in layers: continue
        assert not hasattr(block, "conv_fusion"), f"layer {li} already fused"
        orig = block.conv_tp
        block.conv_tp = FlashConv(orig, dtype)
        block.conv_fusion = True
        saved.append((li, orig))
    return saved


def uninstall(model, saved):
    for li, orig in saved:
        block = model.interactions[li]
        block.conv_tp = orig
        del block.conv_fusion


def is_installed(model):
    return [hasattr(b, "conv_fusion") for b in model.interactions]
