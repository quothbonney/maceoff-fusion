"""Adapter review tests (item d). usage: python -I adapter_review.py <flashtp_tree> <out_json>
Uses the (unmodified) adapter scripts/flash_proto/flashtp_adapter.py with FlashTP from <flashtp_tree>."""
import os, sys, json, io, copy, subprocess, traceback
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
tree, out_path = sys.argv[1:3]
sys.path.insert(0, tree)
import flashTP_e3nn
assert flashTP_e3nn.__file__.startswith(tree), flashTP_e3nn.__file__
R = os.environ["R"]
sys.path.insert(1, f"{R}/scripts/flash_proto")
import numpy as np, torch
from ase import Atoms
from mace.calculators import MACECalculator
import flashtp_adapter as FA
from systems import get_system

calc = MACECalculator(model_paths=f"{R}/models/MACE-OFF23_medium.model", device="cuda", default_dtype="float64",
                      compute_stress=False)
model = calc.models[0]
res = {"slurm_job": os.environ.get("SLURM_JOB_ID"), "flashtp": flashTP_e3nn.__file__,
       "gpu_uuid": subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
       "interaction_classes": [type(b).__name__ for b in model.interactions], "r_max": float(model.r_max), "tests": {}}


def ef(atoms):
    a = atoms.copy(); a.calc = calc; calc.reset()
    return a.get_potential_energy(), a.get_forces().copy()


def cmp(ref, got, nat):
    return dict(dE_per_atom=abs(ref[0] - got[0]) / nat, max_dF=float(np.abs(ref[1] - got[1]).max()),
                E_ref=ref[0], E_flash=got[0], finite=bool(np.isfinite(got[0]) and np.isfinite(got[1]).all()))


def record(name, fn):
    try:
        r = fn(); r["status"] = "ok"
    except Exception as e:
        r = {"status": "exception", "error": f"{type(e).__name__}: {e}"[:500], "tb": traceback.format_exc()[-800:]}
    res["tests"][name] = r
    print(name, json.dumps(r, default=float)[:900], flush=True)


h2o = get_system("H2O"); w192 = get_system("water192")
single = Atoms("O", positions=[[0, 0, 0]])
far = Atoms("OH", positions=[[0, 0, 0], [0, 0, 20.0]])       # > r_max apart -> 0 edges
far_plus = Atoms("OHH", positions=[[0, 0, 0], [0, 0, 20.0], [0, 0.96, 20.0]])  # one isolated atom + 1 pair
perm = w192[np.random.default_rng(3).permutation(len(w192))]  # same edge count, different (src,dst) pairs
systems = {"single_atom": single, "two_atoms_20A": far, "isolated_plus_pair": far_plus, "H2O": h2o,
           "water192": w192, "water192_permuted": perm}
refs = {k: ef(v) for k, v in systems.items()}

# --- state before install
sd0 = {k: v.clone() for k, v in model.state_dict().items()}
n_param0 = sum(1 for _ in model.parameters()); n_buf0 = sum(1 for _ in model.buffers())
orig_mods = [b.conv_tp for b in model.interactions]
saved = FA.install(model)
sd1 = model.state_dict()
res["tests"]["install_state_dict"] = dict(
    keys_added=sorted(set(sd1) - set(sd0))[:20], n_keys_added=len(set(sd1) - set(sd0)),
    keys_removed=sorted(set(sd0) - set(sd1))[:20], n_keys_removed=len(set(sd0) - set(sd1)),
    n_params_before=n_param0, n_params_after=sum(1 for _ in model.parameters()),
    n_buffers_before=n_buf0, n_buffers_after=sum(1 for _ in model.buffers()))
print("install_state_dict", res["tests"]["install_state_dict"], flush=True)

# --- graphs: empty edges, changing graphs (alternating order to catch stale state)
order = ["H2O", "single_atom", "water192", "two_atoms_20A", "water192_permuted", "isolated_plus_pair", "H2O", "water192"]
for i, k in enumerate(order):
    record(f"graph{i}_{k}", lambda k=k: cmp(refs[k], ef(systems[k]), len(systems[k])))


# --- direct FlashConv misuse
blk = model.interactions[0]; fc = blk.conv_tp
N, E = 10, 40
x = torch.randn(N, fc.ftp.i_in1.dim if hasattr(fc.ftp, "i_in1") else blk.conv_tp.ftp.i_in1.dim, device="cuda", dtype=torch.float64)
ea = torch.randn(E, 16, device="cuda", dtype=torch.float64)
wt = torch.randn(E, fc.weight_numel, device="cuda", dtype=torch.float64)
ei = torch.randint(0, N, (2, E), device="cuda")
record("direct_fp64_ok", lambda: {"shape": list(fc(x, ea, wt, ei).shape)})
record("dtype_mismatch_fp32_inputs", lambda: {"shape": list(fc(x.float(), ea.float(), wt.float(), ei).shape)})
record("noncontiguous_edge_index", lambda: {"max_diff_vs_contig": float((fc(x, ea, wt, ei.t().contiguous().t()) - fc(x, ea, wt, ei)).abs().max())})

# --- pickling / deepcopy while installed
def save_installed():
    buf = io.BytesIO(); torch.save(model, buf); n = buf.tell(); buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    return {"bytes": n, "reloaded_types": [type(b.conv_tp).__name__ for b in m2.interactions]}
record("torch_save_whole_model_installed", save_installed)
record("deepcopy_installed", lambda: {"types": [type(b.conv_tp).__name__ for b in copy.deepcopy(model).interactions]})
def jit_script():
    torch.jit.script(model); return {}
record("torchscript_installed", jit_script)

# --- uninstall restores exact state
FA.uninstall(model, saved)
sd2 = model.state_dict()
res["tests"]["uninstall_restore"] = dict(
    same_keys=set(sd2) == set(sd0), values_bitwise=all(torch.equal(sd0[k], sd2[k]) for k in sd0),
    same_module_objects=all(b.conv_tp is m for b, m in zip(model.interactions, orig_mods)),
    conv_fusion_removed=not any(hasattr(b, "conv_fusion") for b in model.interactions))
back = {k: ef(systems[k]) for k in ("H2O", "water192")}
res["tests"]["uninstall_restore"]["E_F_bitwise_equal_ref"] = all(
    back[k][0] == refs[k][0] and np.array_equal(back[k][1], refs[k][1]) for k in back)
print("uninstall_restore", res["tests"]["uninstall_restore"], flush=True)
# double install / uninstall idempotence
s1 = FA.install(model)
record("double_install", lambda: {"second_install": str(FA.install(model))})
FA.uninstall(model, s1)
record("double_uninstall", lambda: (FA.uninstall(model, s1), {})[1])
json.dump(res, open(out_path, "w"), indent=1, default=float)
