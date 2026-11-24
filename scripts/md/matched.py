"""Matched-configuration force agreement. Usage: matched.py <system> <tag> [tag...]
For every saved frame of traj_<tag>: e3nn (x2, separate calc.reset) and FlashTP-fp64cg on exactly that configuration."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from mdlib import *
from ase import Atoms
name, tags = sys.argv[1], sys.argv[2:]
st = np.load(f"{OUT}/start_{name}.npz")
cA, _, _ = make_calc("e3nn"); cB, _, _ = make_calc("flash")
def ev(calc, pos):
    a = Atoms(numbers=st["numbers"], positions=pos, pbc=False); a.calc = calc; calc.reset()
    return a.get_potential_energy(), a.get_forces().copy()
res = dict(env=env_info(), system=name, tags={})
for tag in tags:
    tr = np.load(f"{OUT}/traj_{tag}.npz"); P = tr["positions"]; Fst = tr["forces"]; n = P.shape[1]
    rows = []
    for i, pos in enumerate(P):
        eA, fA = ev(cA, pos); eA2, fA2 = ev(cA, pos); eB, fB = ev(cB, pos)
        d = fB - fA
        rows.append([tr["saved_steps"][i], np.abs(d).max(), np.sqrt((d**2).mean()), abs(eB - eA) / n,
                     np.abs(fA2 - fA).max(), abs(eA2 - eA) / n, np.abs(fA).max(),
                     np.abs(Fst[i] - (fB if "flash" in tag else fA)).max()])
    rows = np.array(rows)
    np.save(f"{OUT}/matched_{tag}.npy", rows)
    s = dict(n_frames=len(rows), max_abs_dF=rows[:, 1].max(), median_max_abs_dF=np.median(rows[:, 1]),
             max_rms_dF=rows[:, 2].max(), max_dE_per_atom=rows[:, 3].max(),
             e3nn_repeat_max_abs_dF=rows[:, 4].max(), e3nn_repeat_max_dE_per_atom=rows[:, 5].max(),
             e3nn_repeat_bitwise_frac=float((rows[:, 4] == 0).mean()), max_abs_F=rows[:, 6].max(),
             stored_vs_reeval_same_backend_max_abs_dF=rows[:, 7].max())
    res["tags"][tag] = s; print(tag, s, flush=True)
json.dump(res, open(f"{OUT}/matched_{name}_{'_'.join(tags)}.json", "w"), indent=1, default=float)
