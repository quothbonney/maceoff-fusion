"""Capture real conv_tp inputs of MACE-OFF23 medium (both interaction layers) for H2O and the
648-atom water cluster (geometry builder copied verbatim from scripts/inspect_mace_layers.py).
CPU, float64 (deterministic). Usage: python -I capture_inputs.py <model> <out.pt>"""
import sys, time, hashlib
import numpy as np, torch
import mace  # sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD for e3nn constants
from ase.build import molecule
from ase import Atoms
from mace.calculators import MACECalculator

model_path, out = sys.argv[1:3]

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

calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64")
model = calc.models[0]; model.eval()
cap = {}
for li, inter in enumerate(model.interactions):
    inter.register_forward_pre_hook(lambda m, a, kw, li=li: cap.setdefault(li, {}).update(
        edge_index=kw["edge_index"].detach().clone(), edge_attrs=kw["edge_attrs"].detach().clone()), with_kwargs=True)
    inter.linear_up.register_forward_hook(lambda m, a, o, li=li: cap[li].update(node_feats=o.detach().clone()))
    inter.conv_tp.register_forward_pre_hook(lambda m, a, li=li: cap[li].update(tp_weights=a[2].detach().clone(),
                                                                                 conv_in1=a[0].detach().clone()))
res = {"meta": {"model": model_path, "torch": torch.__version__}, "systems": {}}
for name, atoms in {"H2O": molecule("H2O"), "water648": water_cluster()}.items():
    cap.clear(); atoms = atoms.copy(); atoms.calc = calc
    t = time.time(); e = atoms.get_potential_energy()
    layers = {}
    for li, c in sorted(cap.items()):
        # sanity: conv_tp in1 is exactly node_feats[edge_index[0]]
        assert torch.equal(c.pop("conv_in1"), c["node_feats"][c["edge_index"][0]])
        assert torch.allclose(c["edge_attrs"], c["edge_attrs"])
        layers[li] = c
        print(name, li, {k: (tuple(v.shape), str(v.dtype)) for k, v in c.items()}, flush=True)
    res["systems"][name] = {"n_atoms": len(atoms), "energy_eV": float(e), "layers": layers}
    print(name, len(atoms), e, f"{time.time()-t:.1f}s", flush=True)
torch.save(res, out)
print("saved", out, hashlib.sha256(open(out, "rb").read()).hexdigest())
