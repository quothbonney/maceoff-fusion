"""Feasibility: try capturing fwd+force-bwd with the STOCK cgpatch FlashTP build (expected to fail). argv: backend"""
import os, sys, json, traceback
R = os.environ["R"]; sys.path.insert(0, f"{R}/scripts/flash_v2")
import hybrid_v2 as HB
sys.path.insert(1, f"{R}/scripts/graph"); sys.path.insert(1, f"{R}/scripts/cudagraph6")
import torch, ase.io
from proto_graph import ProtoBuilder
from graphed_model import GraphedMACE, round_cap
bk = sys.argv[1]
conv, sc = dict(e3nn=("e3nn", "e3nn"), flash=("flashtp", "e3nn"), hybrid=("flashtp", "cueq"))[bk]
calc, _ = HB.build_calc(conv=conv, sc=sc, check_indices=False)
at = ase.io.read(f"{R}/results/workloads/systems/aspirin.extxyz")
bd = ProtoBuilder(calc).build(at)
r = dict(backend=bk, flashTP=HB.flashTP_e3nn.__file__)
try:
    G = GraphedMACE(calc.models[0], bd, round_cap(bd["edge_index"].shape[1]), calc.r_max)
    G.step(bd); torch.cuda.synchronize(); r["capture"] = "no exception"; r["E_graphed"] = float(G.out_E)
    Fg = G.out_F.detach().clone()
    from mace.tools import torch_tools
    with torch_tools.default_dtype("float64"):
        o = calc.models[0](bd, compute_force=True, training=False)
    r["E_eager"] = float(o["energy"]); r["max_dF_graphed_vs_eager"] = float((o["forces"] - Fg).abs().max())
    r["graphed_matches_eager"] = abs(r["E_eager"] - r["E_graphed"]) < 1e-9 and r["max_dF_graphed_vs_eager"] < 1e-12
except Exception as ex:
    r["capture"] = "failed"; r["error"] = f"{type(ex).__name__}: {ex}"[:2000]; traceback.print_exc()
print("FEAS", json.dumps(r)); json.dump(r, open(f"{R}/results/cudagraph6/feas_stock_{bk}.json", "w"), indent=1)
