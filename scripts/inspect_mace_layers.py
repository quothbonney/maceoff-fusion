"""READ-ONLY inventory of MACE-OFF23 medium interaction layers for a FlashTP (uvu_TP) swap.

Does not modify MACE, the venv, or FlashTP. Runs on CPU (no FlashTP kernel needed).
Usage: python -I inspect_mace_layers.py <model_path> <mace_src_root> <flashtp_root> <out_json>
"""
import json, os, sys, time, inspect, hashlib, platform, warnings
import numpy as np
import torch
import mace  # must precede e3nn: mace/__init__.py sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (e3nn constants.pt, trusted checkpoint)
from e3nn import o3
import e3nn, ase
from ase.build import molecule
from ase import Atoms
from mace.calculators import MACECalculator
from mace.tools.scatter import scatter_sum

model_path, src_root, ftp_root, out_json = sys.argv[1:5]
torch.manual_seed(0)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

# ---------------------------------------------------------------- source call sites
def find_lines(path, needles, start_pat=None, end_pat=None):
    lines = open(path).read().splitlines()
    lo, hi = 0, len(lines)
    if start_pat:
        lo = next(i for i, l in enumerate(lines) if start_pat in l)
    if end_pat:
        hi = next(i for i, l in enumerate(lines[lo + 1:], lo + 1) if end_pat in l)
    out = []
    for i in range(lo, hi):
        if any(n in lines[i] for n in needles):
            out.append({"file": os.path.relpath(path, src_root), "line": i + 1, "text": lines[i].strip()})
    return out

blocks = os.path.join(src_root, "mace/modules/blocks.py")
call_needles = ["sc = self.skip_tp(", "node_feats = self.linear_up(", "tp_weights = self.conv_tp_weights(",
                "tp_weights = tp_weights * cutoff", 'if hasattr(self, "conv_fusion")',
                "message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)",
                "mji = self.conv_tp(", "node_feats[edge_index[0]], edge_attrs, tp_weights",
                "message = scatter_sum(", "src=mji, index=edge_index[1]",
                "message = self.linear(message) / self.avg_num_neighbors",
                "message = self.skip_tp(message, node_attrs)", "self.reshape(message)",
                "message = self.truncate_ghosts(message, n_real)"]
call_sites = {
    "RealAgnosticInteractionBlock.forward": find_lines(blocks, call_needles,
        "class RealAgnosticInteractionBlock(", "class RealAgnosticResidualInteractionBlock("),
    "RealAgnosticResidualInteractionBlock.forward": find_lines(blocks, call_needles,
        "class RealAgnosticResidualInteractionBlock(", "class RealAgnosticDensityInteractionBlock("),
    "conv_tp construction (both blocks)": find_lines(blocks, ["self.conv_tp = TensorProduct(",
        "irreps_mid, instructions = tp_out_irreps_with_instructions("]),
    "edge vectors (sender/receiver)": find_lines(os.path.join(src_root, "mace/modules/utils.py"),
        ["sender = edge_index[0]", "receiver = edge_index[1]", "vectors = positions[receiver] - positions[sender]"],
        "def get_edge_vectors_and_lengths", "def _check_non_zero"),
    "edge_index construction": find_lines(os.path.join(src_root, "mace/data/neighborhood.py"),
        ["edge_index = np.stack((sender, receiver))"]) +
        find_lines(os.path.join(src_root, "mace/data/atomic_data.py"), ["edge_index=torch.tensor(edge_index, dtype=torch.long)"]),
    "interaction call in ScaleShiftMACE/MACE forward": find_lines(os.path.join(src_root, "mace/modules/models.py"),
        ["node_feats, sc = interaction(", "edge_index=data[\"edge_index\"],", "cutoff=cutoff,"], None, None)[:6],
    "tp_out_irreps_with_instructions": find_lines(os.path.join(src_root, "mace/modules/irreps_tools.py"),
        ['instructions.append((i, j, k, "uvu", trainable))', "irreps_out, permut, _ = irreps_out.sort()",
         "instructions = sorted(instructions, key=lambda x: x[2])"]),
    "reshape_irreps layout (no cueq -> mul_ir)": find_lines(os.path.join(src_root, "mace/modules/irreps_tools.py"),
        ["field = field.reshape(batch, mul, d)", "return torch.cat(out, dim=-1)"], "class reshape_irreps", "def mask_head"),
    "wrapper_ops.TensorProduct fallback to e3nn": find_lines(os.path.join(src_root, "mace/modules/wrapper_ops.py"),
        ["CUET_AVAILABLE", "OEQ_AVAILABLE", "return o3.TensorProduct("], "class TensorProduct:", "class FullyConnectedTensorProduct"),
    "scatter_sum impl": find_lines(os.path.join(src_root, "mace/tools/scatter.py"), ["return out.scatter_add_(dim, index, src)"]),
}

ftp = os.path.join(ftp_root, "flashTP_e3nn")
ftp_large = os.path.join(ftp, "sptp_exp_opt_large/fused_e3nn_exp_opt_large.py")
flashtp_refs = {
    "uvu_TP signature": find_lines(os.path.join(ftp, "flashTP.py"), ["def uvu_TP(", "    irreps_in1,", "    instructions,",
        "block_batch_cnt=None", "smem_size = None", 'device="cuda"', "dtype=torch.float32", "use_lammps=False"], "def uvu_TP(", '"""'),
    "uvu_TP builds e3nn TP w/ defaults": find_lines(os.path.join(ftp, "flashTP.py"), ["tp = e3nn.o3.TensorProduct(",
        "shared_weights=False,", "internal_weights=False,", "# irrep_normalization", "# path_normalization",
        "if len(unique_cg_val) <= 256:", "cg_fp32 = cg.to(torch.float32)", "e3nn.o3.wigner_3j("]),
    "asserts / layout assumptions (large kernel)": find_lines(ftp_large, ["assert", "tp_inst_outorder = sorted(",
        "fiber_start = per_path_fiber_start[out_ir_idx]", "out2weight_order", "node_cnt = in1.shape[0]",
        "out = torch.zeros((node_cnt, out_size)", "dtype=torch.uint8", "dtype=torch.uint16", "torch.cuda.get_device_name()",
        "def forward(self, in1, in2, weight, per_edge_src, per_edge_dst)"]),
    "C++ input checks": find_lines(os.path.join(ftp, "sptp_exp_opt_large/sptp_exp_opt_large.cpp"),
        ["#define CHECK_CUDA", "#define CHECK_CONTIGUOUS", "CHECK_INPUT(per_edge_src)", "kFloat64", "else assert(false)"])[:8],
    "CUDA kernel edge index type / scatter": find_lines(os.path.join(ftp, "sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp_double.cu"),
        ["const int* __restrict__ per_edge_src", "const int src_idx = per_edge_src", "const int dst_idx = per_edge_dst",
         "atomicAdd(out+out_idx", "per_edge_src.data<int>()", "shmem_scratch[shmem_idx] = my_shmem_uvuv[i] * reg_w_path_norm"]),
    "reference usage in example": find_lines(os.path.join(ftp_root, "example/flashTP_test.py"), ["in1 = in1_node[edge_src]",
        "out_large_exp = tp(in1,in2,weight)", "flashtp = flashTP_e3nn.uvu_TP(", "out_ours = flashtp(",
        "dtype=torch.int32)", "scatter_add_("]),
}

# ---------------------------------------------------------------- load model (CPU, read-only)
log("loading", model_path)
calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64")
model = calc.models[0]
model.eval()
param_dtype = next(model.parameters()).dtype
log("model", type(model).__name__, param_dtype, "r_max", float(model.r_max))
default_dtype_after_calc = str(torch.get_default_dtype())
# Build all reference e3nn objects / wigner_3j in the model dtype (as flashTP example does via torch.set_default_dtype)
torch.set_default_dtype(param_dtype)

def irreps_str(x): return str(o3.Irreps(x))

# ------------------------------------------------------------ FlashTP metadata emulation (pure python)
def flashtp_emulation(i_in1, i_in2, i_out, e3nn_insts, dtype):
    """Re-runs the *python-side* bookkeeping of flashTP.uvu_TP/extract_cg_info and
    fused_uvu_TP_exp_opt_large.cgmat2fiber/metadata_gen (no CUDA) to see whether it
    would assert, and what per-path weight offsets / path weights it would use."""
    res = {}
    unique_cg, unique_cg_mat = [], {}
    for ins in e3nn_insts:
        l1, l2, l3 = i_in1[ins.i_in1].ir.l, i_in2[ins.i_in2].ir.l, i_out[ins.i_out].ir.l
        cg = o3.wigner_3j(l1, l2, l3)
        unique_cg += cg.to(torch.float32).unique().to(dtype).tolist()
        unique_cg_mat[f"{l1}_{l2}_{l3}"] = cg
    ucg = list(set(unique_cg))
    res["n_unique_cg"] = len(ucg)
    res["kernel_variant"] = "fused_uvu_TP_exp_opt_large" if len(ucg) <= 256 else "fused_uvu_TP_exp_opt_extcg"
    outorder = sorted(e3nn_insts, key=lambda x: x.i_out)
    res["i_out_is_permutation_of_range"] = sorted(x.i_out for x in e3nn_insts) == list(range(len(i_out)))
    res["one_instruction_per_output_irrep"] = len(set(x.i_out for x in e3nn_insts)) == len(e3nn_insts) == len(i_out)
    fiber_start, fiber, per_in1 = [0], [], {}
    lookup_ok = True
    for ins in outorder:
        cg = unique_cg_mat[f"{i_in1[ins.i_in1].ir.l}_{i_in2[ins.i_in2].ir.l}_{i_out[ins.i_out].ir.l}"]
        for i, j, k in cg.nonzero():
            try:
                idx = ucg.index(cg[i, j, k].to(torch.float32))
            except ValueError:
                lookup_ok = False; idx = -1
            fiber.append([int(i), int(j), int(k), idx])
        fiber_start.append(len(fiber))
        per_in1.setdefault(ins.i_in1, []).append([ins.i_out, ins.i_in2, ins.path_weight])
    res["cg_value_lookup_ok"] = lookup_ok
    res["total_fiber_nnz"] = len(fiber)
    res["fiber_nnz_fits_uint16"] = len(fiber) < 65536
    res["fiber_ijk_fit_uint8"] = all(max(f[:3]) < 256 for f in fiber) and all(0 <= f[3] < 256 for f in fiber)
    # metadata_gen weight bookkeeping
    chunks, cur, out_order = [], 0, []
    for ins in e3nn_insts:
        u, v = i_in1[ins.i_in1].mul, i_in2[ins.i_in2].mul
        chunks.append((cur, cur + u * v)); cur += u * v; out_order.append(ins.i_out)
    out2w = torch.tensor(out_order).argsort().tolist()
    mul_ok = []
    in2_slices = i_in2.slices()
    per_path = []
    for k1, (mul, ir) in enumerate(i_in1):
        mul_ok.append(mul % 32 == 0)
        for o, j, pw in per_in1.get(k1, []):
            per_path.append({"i_in1": k1, "i_in2": j, "i_out": o, "path_weight": pw,
                             "flashtp_weight_offset": chunks[out2w[o]][0]})
    res["metadata_gen_assert_mul_mod_32_ok"] = all(mul_ok)
    res["every_in1_irrep_has_paths"] = all(k in per_in1 for k in range(len(i_in1)))
    res["in2_slice_stop_fits_uint8"] = max(s.stop for s in in2_slices) < 256
    res["out_ir_dim_fits_uint8"] = max(ir.dim for _, ir in i_out) < 256
    res["upath_cnt(per_exec_info)"] = sum(m // 32 for m, _ in i_in1)
    res["per_path"] = per_path
    return res

def flashtp_math_emulation(i_in1, i_in2, i_out, per_path, x_node, y, w, src, dst, cg_round_fp32=False):
    """Dense torch re-statement of what the FlashTP fwd kernel computes:
    out[dst] += path_weight * sum_ijk x[src,u,i] y[j] cg[i,j,k] * W[e, offset + u] (uvu, mul_in2=1)."""
    out = torch.zeros(x_node.shape[0], i_out.dim, dtype=x_node.dtype)
    s1, s2, so = i_in1.slices(), i_in2.slices(), i_out.slices()
    x = x_node[src]
    for p in per_path:
        mu, ir1 = i_in1[p["i_in1"]]; mv, ir2 = i_in2[p["i_in2"]]; mo, ir3 = i_out[p["i_out"]]
        assert mv == 1
        cg = o3.wigner_3j(ir1.l, ir2.l, ir3.l)
        # flashTP.extract_cg_info: unique_cg += cg.to(float32).unique().to(dtype) -> kernel CG table is fp32-rounded
        cg = (cg.to(torch.float32) if cg_round_fp32 else cg).to(x.dtype)
        xa = x[:, s1[p["i_in1"]]].reshape(-1, mu, ir1.dim)
        ya = y[:, s2[p["i_in2"]]]
        wa = w[:, p["flashtp_weight_offset"]: p["flashtp_weight_offset"] + mu * mv]
        z = p["path_weight"] * torch.einsum("eui,ej,ijk,eu->euk", xa, ya, cg, wa)
        out_slice = torch.zeros(x_node.shape[0], mo * ir3.dim, dtype=x.dtype).index_add_(0, dst, z.reshape(z.shape[0], -1))
        out[:, so[p["i_out"]]] += out_slice
    return out

# ---------------------------------------------------------------- static per-layer inventory
layers = []
for li, inter in enumerate(model.interactions):
    tp = inter.conv_tp
    insts = [{"i_in1": i.i_in1, "i_in2": i.i_in2, "i_out": i.i_out, "connection_mode": i.connection_mode,
              "has_weight": i.has_weight, "path_weight": float(i.path_weight), "path_shape": list(i.path_shape),
              "in1": str(tp.irreps_in1[i.i_in1]), "in2": str(tp.irreps_in2[i.i_in2]), "out": str(tp.irreps_out[i.i_out])}
             for i in tp.instructions]
    mlp = inter.conv_tp_weights
    mlp_layers = [list(p.shape) for n, p in mlp.named_parameters()]
    d = {
        "layer": li,
        "interaction_class": f"{type(inter).__module__}.{type(inter).__name__}",
        "conv_tp_class": f"{type(tp).__module__}.{type(tp).__name__}",
        "has_conv_fusion_attr": hasattr(inter, "conv_fusion"),
        "cueq_config": repr(getattr(inter, "cueq_config", "<absent>")),
        "oeq_config": repr(getattr(inter, "oeq_config", "<absent>")),
        "avg_num_neighbors": float(inter.avg_num_neighbors),
        "node_feats_irreps": irreps_str(inter.node_feats_irreps),
        "edge_irreps": irreps_str(getattr(inter, "edge_irreps", inter.node_feats_irreps)),
        "edge_attrs_irreps": irreps_str(inter.edge_attrs_irreps),
        "edge_feats_irreps": irreps_str(inter.edge_feats_irreps),
        "target_irreps": irreps_str(inter.target_irreps),
        "hidden_irreps": irreps_str(inter.hidden_irreps),
        "linear_up": {"class": type(inter.linear_up).__name__, "irreps_in": irreps_str(inter.linear_up.irreps_in),
                      "irreps_out": irreps_str(inter.linear_up.irreps_out), "weight_numel": inter.linear_up.weight_numel},
        "conv_tp_weights": {"class": f"{type(mlp).__module__}.{type(mlp).__name__}", "hs": list(mlp.hs),
                            "param_shapes": mlp_layers, "act": repr(getattr(mlp, "act", None))},
        "linear": {"class": type(inter.linear).__name__, "irreps_in": irreps_str(inter.linear.irreps_in),
                   "irreps_out": irreps_str(inter.linear.irreps_out), "weight_numel": inter.linear.weight_numel},
        "skip_tp": {"class": type(inter.skip_tp).__name__, "irreps_in1": irreps_str(inter.skip_tp.irreps_in1),
                    "irreps_in2": irreps_str(inter.skip_tp.irreps_in2), "irreps_out": irreps_str(inter.skip_tp.irreps_out)},
        "reshape_irreps": {"irreps": irreps_str(inter.reshape.irreps),
                           "cueq_config": repr(getattr(inter.reshape, "cueq_config", "<absent>"))},
        "conv_tp": {
            "irreps_in1": str(tp.irreps_in1), "irreps_in2": str(tp.irreps_in2), "irreps_out": str(tp.irreps_out),
            "irreps_in1_dim": tp.irreps_in1.dim, "irreps_in2_dim": tp.irreps_in2.dim, "irreps_out_dim": tp.irreps_out.dim,
            "instructions": insts, "n_instructions": len(insts), "weight_numel": tp.weight_numel,
            "shared_weights": tp.shared_weights, "internal_weights": tp.internal_weights,
            "irrep_normalization": getattr(tp, "irrep_normalization", "not stored by e3nn 0.4.4; default 'component' (no kwarg passed by wrapper_ops)"),
            "path_normalization": getattr(tp, "path_normalization", "not stored by e3nn 0.4.4; default 'element' (no kwarg passed by wrapper_ops)"),
            "in1_var": list(map(float, tp.in1_var)) if hasattr(tp, "in1_var") else None,
            "in2_var": list(map(float, tp.in2_var)) if hasattr(tp, "in2_var") else None,
            "out_var": list(map(float, tp.out_var)) if hasattr(tp, "out_var") else None,
            "irreps_out_is_sorted": str(tp.irreps_out.sort().irreps) == str(tp.irreps_out),
            "irreps_out_is_simplified": str(tp.irreps_out.simplify()) == str(tp.irreps_out),
            "instructions_sorted_by_i_out": [i.i_out for i in tp.instructions] == sorted(i.i_out for i in tp.instructions),
        },
        "checks": {
            "all_uvu": all(i.connection_mode == "uvu" for i in tp.instructions),
            "all_has_weight": all(i.has_weight for i in tp.instructions),
            "irreps_in1_muls": [m for m, _ in tp.irreps_in1],
            "irreps_out_muls": [m for m, _ in tp.irreps_out],
            "irreps_in2_muls": [m for m, _ in tp.irreps_in2],
            "in1_muls_mult_of_32": all(m % 32 == 0 for m, _ in tp.irreps_in1),
            "out_muls_mult_of_32": all(m % 32 == 0 for m, _ in tp.irreps_out),
            "in2_all_mul_1": all(m == 1 for m, _ in tp.irreps_in2),
            "weight_numel_eq_sum_u*v": tp.weight_numel == sum(tp.irreps_in1[i.i_in1].mul * tp.irreps_in2[i.i_in2].mul for i in tp.instructions),
            "mlp_out_eq_weight_numel": mlp.hs[-1] == tp.weight_numel,
        },
    }
    # Check MACE conv_tp == e3nn TP built exactly the way flashTP.uvu_TP builds it (default normalization).
    inst_tuples = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]
    d["flashtp_inst_tuple"] = inst_tuples
    ref = o3.TensorProduct(tp.irreps_in1, tp.irreps_in2, tp.irreps_out, inst_tuples,
                           shared_weights=False, internal_weights=False).to(param_dtype)
    d["flashtp_reference_tp"] = {
        "normalization_note": "e3nn 0.4.4 TensorProduct does not store irrep/path_normalization; they are folded into Instruction.path_weight (e3nn _tensor_product.py:213-217 defaults component/element, :289-314). Neither MACE wrapper_ops (return o3.TensorProduct(...) w/o kwargs) nor flashTP.uvu_TP pass them, so both use defaults; equality of path_weight below is the operative check.",
        "path_weights_equal_mace": all(abs(a.path_weight - b.path_weight) == 0 for a, b in zip(ref.instructions, tp.instructions)),
        "instructions_equal_mace": [(a.i_in1, a.i_in2, a.i_out, a.connection_mode, a.has_weight, a.path_shape) for a in ref.instructions]
                                   == [(b.i_in1, b.i_in2, b.i_out, b.connection_mode, b.has_weight, b.path_shape) for b in tp.instructions],
        "weight_numel_equal": ref.weight_numel == tp.weight_numel,
    }
    d["flashtp_metadata_emulation"] = flashtp_emulation(tp.irreps_in1, tp.irreps_in2, tp.irreps_out, tp.instructions, param_dtype)
    d["_ref_tp"] = ref
    layers.append(d)
    log(f"layer {li}: {d['interaction_class']} conv_tp {tp.irreps_in1} x {tp.irreps_in2} -> {tp.irreps_out} "
        f"n_inst={len(insts)} weight_numel={tp.weight_numel}")

# layer-to-layer difference
L0, L1 = layers[0]["conv_tp"], layers[1]["conv_tp"]
same = (L0["irreps_in1"], L0["irreps_in2"], L0["irreps_out"], L0["instructions"]) == \
       (L1["irreps_in1"], L1["irreps_in2"], L1["irreps_out"], L1["instructions"])

# ---------------------------------------------------------------- systems
def water_cluster(n_side=6, spacing=3.1, seed=0):
    rng = np.random.default_rng(seed)
    w = molecule("H2O"); base = w.positions - w.positions.mean(0)
    pos, sym = [], []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, dd = q
                Rm = np.array([[a*a+b*b-c*c-dd*dd, 2*(b*c-a*dd), 2*(b*dd+a*c)],
                               [2*(b*c+a*dd), a*a-b*b+c*c-dd*dd, 2*(c*dd-a*b)],
                               [2*(b*dd-a*c), 2*(c*dd+a*b), a*a-b*b-c*c+dd*dd]])
                pos.append(base @ Rm.T + spacing * np.array([i, j, k]) + rng.normal(scale=0.1, size=3))
                sym += w.get_chemical_symbols()
    return Atoms(symbols=sym, positions=np.concatenate(pos), pbc=False)

systems = {"H2O": molecule("H2O"), "water_cluster_6x6x6": water_cluster()}

# ---------------------------------------------------------------- hooks
cap = {}
def shp(t): return {"shape": list(t.shape), "dtype": str(t.dtype), "contiguous": bool(t.is_contiguous()),
                    "stride": list(t.stride())}
handles = []
for li, inter in enumerate(model.interactions):
    def pre_inter(mod, args, kwargs, li=li):
        ei = kwargs["edge_index"]
        cap[li] = {"edge_index": ei.detach().clone(), "node_feats_in": shp(kwargs["node_feats"]),
                   "edge_attrs_t": kwargs["edge_attrs"].detach().clone(), "edge_feats": shp(kwargs["edge_feats"]),
                   "cutoff": None if kwargs.get("cutoff") is None else shp(kwargs["cutoff"]),
                   "kwargs_keys": sorted(kwargs.keys())}
    def post_inter(mod, args, kwargs, out, li=li):
        cap[li]["interaction_out"] = shp(out[0]); cap[li]["sc"] = None if out[1] is None else shp(out[1])
    def post_linear_up(mod, args, out, li=li):
        cap[li]["linear_up_out_t"] = out.detach().clone()
    def pre_conv(mod, args, li=li):
        cap[li]["conv_in1"] = shp(args[0]); cap[li]["conv_in2"] = shp(args[1]); cap[li]["conv_w"] = shp(args[2])
        cap[li]["conv_n_args"] = len(args)
        cap[li]["tp_weights_t"] = args[2].detach().clone()
    def post_conv(mod, args, out, li=li):
        cap[li]["mji"] = shp(out); cap[li]["mji_t"] = out.detach().clone()
    def pre_linear(mod, args, li=li):
        cap[li]["message_post_scatter"] = shp(args[0]); cap[li]["message_post_scatter_t"] = args[0].detach().clone()
    def post_linear(mod, args, out, li=li):
        cap[li]["linear_out"] = shp(out)
    def post_reshape(mod, args, out, li=li):
        cap[li]["reshape_in"] = shp(args[0]); cap[li]["reshape_out"] = shp(out)
    handles += [inter.register_forward_pre_hook(pre_inter, with_kwargs=True),
                inter.register_forward_hook(post_inter, with_kwargs=True),
                inter.linear_up.register_forward_hook(post_linear_up),
                inter.conv_tp.register_forward_pre_hook(pre_conv),
                inter.conv_tp.register_forward_hook(post_conv),
                inter.linear.register_forward_pre_hook(pre_linear),
                inter.linear.register_forward_hook(post_linear),
                inter.reshape.register_forward_hook(post_reshape)]

MAX_CHECK_EDGES = 4096
runtime = {}
for name, atoms in systems.items():
    cap.clear()
    atoms = atoms.copy(); atoms.calc = calc
    ts = time.time(); e = atoms.get_potential_energy(); dt = time.time() - ts
    log(f"{name}: n_atoms={len(atoms)} E={e:.6f} eV ({dt:.1f}s CPU)")
    per = {"n_atoms": len(atoms), "energy_eV": float(e), "cpu_seconds": dt,
           "extent_A": (atoms.positions.max(0) - atoms.positions.min(0)).tolist(), "layers": {}}
    for li in sorted(cap):
        c = cap[li]; ei = c["edge_index"]; n_nodes = c["linear_up_out_t"].shape[0]; n_edges = ei.shape[1]
        src, dst = ei[0], ei[1]
        sorted_by_dst = bool((dst[1:] >= dst[:-1]).all()) if n_edges > 1 else True
        sorted_by_src = bool((src[1:] >= src[:-1]).all()) if n_edges > 1 else True
        indeg = torch.bincount(dst, minlength=n_nodes).double()
        lr = {k: v for k, v in c.items() if not k.endswith("_t") and k != "edge_index"}
        lr.update({
            "n_nodes": n_nodes, "n_edges": n_edges,
            "edge_index": {**shp(ei), "row0_role": "sender (gathered: node_feats[edge_index[0]])",
                           "row1_role": "receiver (scatter index)", "sorted_by_receiver(row1)": sorted_by_dst,
                           "sorted_by_sender(row0)": sorted_by_src, "row0_contiguous": bool(src.is_contiguous()),
                           "max_index": int(ei.max()) if n_edges else -1, "fits_int32": int(ei.max()) < 2**31 if n_edges else True,
                           "has_self_loops": bool((src == dst).any())},
            "edge_attrs": shp(c["edge_attrs_t"]), "node_feats_into_gather(linear_up_out)": shp(c["linear_up_out_t"]),
            "avg_neighbors_observed(n_edges/n_nodes)": n_edges / n_nodes,
            "in_degree_min_max_mean": [float(indeg.min()), float(indeg.max()), float(indeg.mean())],
            "model_avg_num_neighbors": layers[li]["avg_num_neighbors"],
        })
        # numerical checks (CPU, float64)
        m = min(n_edges, MAX_CHECK_EDGES)
        x_node = c["linear_up_out_t"]; y = c["edge_attrs_t"][:m]; w = c["tp_weights_t"][:m]
        s_, d_ = src[:m], dst[:m]
        with torch.no_grad():
            ref = layers[li]["_ref_tp"]
            mji_ref = ref(x_node[s_], y, w)
            emu = flashtp_math_emulation(model.interactions[li].conv_tp.irreps_in1, model.interactions[li].conv_tp.irreps_in2,
                                         model.interactions[li].conv_tp.irreps_out,
                                         layers[li]["flashtp_metadata_emulation"]["per_path"], x_node, y, w, s_, d_)
            emu32 = flashtp_math_emulation(model.interactions[li].conv_tp.irreps_in1, model.interactions[li].conv_tp.irreps_in2,
                                         model.interactions[li].conv_tp.irreps_out,
                                         layers[li]["flashtp_metadata_emulation"]["per_path"], x_node, y, w, s_, d_, cg_round_fp32=True)
            msg_mace_subset = scatter_sum(src=c["mji_t"][:m], index=d_, dim=0, dim_size=n_nodes)
            msg_full = scatter_sum(src=c["mji_t"], index=dst, dim=0, dim_size=n_nodes)
        lr["numerics"] = {
            "edges_checked": m,
            "max_abs_mji_mace_vs_e3nn_default_TP(as_uvu_TP_builds)": float((mji_ref - c["mji_t"][:m]).abs().max()),
            "max_abs_scatter_full_vs_captured_linear_input": float((msg_full - c["message_post_scatter_t"]).abs().max()),
            "max_abs_flashtp_math_emulation_vs_mace_gather_tp_scatter": float((emu - msg_mace_subset).abs().max()),
            "max_abs_flashtp_emulation_with_fp32_rounded_cg_table_vs_mace": float((emu32 - msg_mace_subset).abs().max()),
            "ref_scale_max_abs_message": float(msg_mace_subset.abs().max()),
        }
        per["layers"][li] = lr
        log(f"  L{li}: nodes={n_nodes} edges={n_edges} avg_nb={n_edges/n_nodes:.2f} (model {layers[li]['avg_num_neighbors']:.3f}) "
            f"in1={c['conv_in1']['shape']} in2={lr['edge_attrs']['shape']} w={c['conv_w']['shape']} mji={c['mji']['shape']} "
            f"msg={c['message_post_scatter']['shape']} out={c['interaction_out']['shape']} numerics={lr['numerics']}")
    runtime[name] = per
for h in handles: h.remove()

for d in layers: d.pop("_ref_tp")
result = {
    "meta": {"host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__, "e3nn": e3nn.__version__,
             "mace": mace.__version__, "mace_imported_from": mace.__file__, "ase": ase.__version__,
             "model_path": model_path, "model_sha256": sha(model_path), "model_class": type(model).__name__,
             "model_dtype": str(param_dtype), "torch_default_dtype_after_MACECalculator": default_dtype_after_calc, "torch_default_dtype_set_by_script_for_reference_objects": str(param_dtype),
             "r_max": float(model.r_max), "num_interactions": len(model.interactions),
             "spherical_harmonics": {"irreps_out": str(model.spherical_harmonics.irreps_out),
                                     "normalize": getattr(model.spherical_harmonics, "normalize", None),
                                     "normalization": getattr(model.spherical_harmonics, "normalization", None)},
             "cuequivariance_importable": False, "openequivariance_importable": False, "device": "cpu"},
    "layers": layers,
    "layer0_vs_layer1_conv_tp_identical": same,
    "needs_separate_flashtp_instances": not same,
    "call_sites_mace_v0.3.17": call_sites,
    "flashtp_source_refs": flashtp_refs,
    "systems": runtime,
}
for mod in ("cuequivariance", "openequivariance"):
    try:
        __import__(mod); result["meta"][f"{mod}_importable"] = True
    except Exception:
        pass
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(result, open(out_json, "w"), indent=1, default=str)
log("wrote", out_json)
print(json.dumps({"layers": [{k: v for k, v in d.items() if k in ("layer", "interaction_class", "conv_tp_class", "checks",
       "flashtp_reference_tp", "has_conv_fusion_attr")} for d in layers], "identical": same}, indent=1, default=str))
