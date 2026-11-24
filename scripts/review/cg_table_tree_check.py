"""CPU check of the CG table each FlashTP tree feeds the kernels (review items a, b).

usage: python -I cg_table_tree_check.py <tree> <label>
For each config/layer and each (torch default dtype, requested dtype) combination, run the tree's own
extract_cg_info + unique-table construction + the tree's cgmat2fiber lookup rule, then reconstruct the
CG value the kernel would use for every nonzero entry and compare with e3nn's exact fp64 wigner_3j.
Also emits a fingerprint of the fp32 table/fiber indices so trees can be compared for bitwise equality.
"""
import os, sys, json, hashlib, struct
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
tree, label = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import torch, e3nn.o3 as o3
import flashTP_e3nn
from flashTP_e3nn import flashTP as F
assert flashTP_e3nn.__file__.startswith(tree), flashTP_e3nn.__file__
R = os.environ["R"]

try:
    from flashTP_e3nn.cg_table import build_unique_cg, cg_index  # exp/fp64-cg tree
    variant = "fp64cg"
except ImportError:
    variant = "plain"
src_large = open(f"{tree}/flashTP_e3nn/sptp_exp_opt_large/fused_e3nn_exp_opt_large.py").read()


def make_table(unique_cg, dtype):
    if variant == "fp64cg":
        return build_unique_cg(unique_cg, dtype)
    return list(set(unique_cg))


def lookup(table, x, dtype):
    if variant == "fp64cg":
        return cg_index(table, x, dtype)
    if ".to(self.dtype))" in src_large:
        return table.index(x.to(dtype))
    return table.index(x.to(torch.float32))


def load_cfg(fn):
    out = []
    for line in open(fn).read().split("\n"):
        if line.strip():
            tp = json.loads(line)["tp"]
            out.append((o3.Irreps(tp[0]), o3.Irreps(tp[1]), o3.Irreps(tp[2]), [tuple(x) for x in tp[3]]))
    return out


cases = [("mace/medium.json", 0), ("mace/medium.json", 1), ("sevennet/sevennet-l3i5.json", 1),
         ("nequip/4_32_4_p_sc.txt", 1), ("nequip/4_32_5_p_sc.txt", 1)]
res = {"tree": tree, "label": label, "variant": variant, "file": flashTP_e3nn.__file__, "cases": {}}
for fn, li in cases:
    i1, i2, io, inst = load_cfg(f"{R}/src/flashTP/ir_config/{fn}")[li]
    for default in (torch.float32, torch.float64):
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            tp = o3.TensorProduct(i1, i2, io, inst, shared_weights=False, internal_weights=False)
            ucg, mats = F.extract_cg_info(i1, i2, io, tp.instructions, dtype)
            table = make_table(ucg, dtype)
            t_table = torch.tensor(table, dtype=dtype)  # what the module uploads
            max_err, n_fib, h = 0.0, 0, hashlib.sha256()
            h.update(t_table.double().numpy().tobytes())
            for key, m in mats.items():
                l1, l2, l3 = map(int, key.split("_"))
                exact = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
                for i, j, k in m.nonzero():
                    idx = lookup(table, m[i, j, k], dtype)
                    max_err = max(max_err, abs(float(t_table[idx]) - float(exact[i, j, k])))
                    n_fib += 1
                    h.update(key.encode()); h.update(struct.pack("<iiii", int(i), int(j), int(k), idx))
            torch.set_default_dtype(torch.float32)
            tag = f"{fn}:L{li}|default={str(default)[6:]}|dtype={str(dtype)[6:]}"
            res["cases"][tag] = dict(n_table=len(table), n_fibers=n_fib, max_abs_cg_err=max_err,
                                     table_dtype=str(t_table.dtype), fingerprint=h.hexdigest()[:16])
            print(label, tag, res["cases"][tag], flush=True)
json.dump(res, open(f"{R}/results/review/cg_table_tree_{label}.json", "w"), indent=1)
