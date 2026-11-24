"""prof8 config builder. cfg = <kind>[+embed][+node]; kind in {e3nn, flash, hybrid (FlashTP conv + cueq SC),
nodehybrid (node-centric conv [C nodeconv.install] + cueq SC)}. Import AFTER graphed_runner."""
import os, sys
R = os.environ["R"]
import graphed_runner as GR

NODE_FWD = os.environ.get("NODE_FWD", "auto")


def _tr(parts):
    tr = []
    if "embed" in parts:
        sys.path.insert(0, f"{R}/scripts/fuse7_embed"); import fused_embed as FE; tr.append(FE.install)
    if "node" in parts:
        sys.path.insert(0, f"{R}/scripts/fuse7_node"); import fused_node as FN; tr.append(FN.install)
    return tr


def build(cfg):
    parts = cfg.split("+"); kind = parts[0]
    if kind != "nodehybrid":
        return GR.build_backend(kind, transforms=_tr(parts))
    sys.path.insert(0, f"{R}/scripts/nodetp8_bwd"); import nodeconv as NC
    calc, info = GR.HB.build_calc(conv="e3nn", sc="cueq", check_indices=False)
    model = calc.models[0]
    handles = [NC.install(model, fwd=NODE_FWD)]
    for t in _tr(parts):
        handles.append(t(model))
    info["backend_map"] = GR.HB.describe(model)
    info["nodeconv"] = [repr(b.conv_tp)[:300] for b in model.interactions]
    calc.graphed7_kind = kind; calc.graphed7_info = info; calc.graphed7_transform_handles = handles
    return calc, model
