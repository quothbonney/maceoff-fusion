"""Backend construction for the workload study. BACKEND_SET=v1 (default until results/flash_v2/READY exists):
  e3nn   : MACECalculator(default_dtype=float64)
  cueq   : MACECalculator(..., enable_cueq=True)
  flash64: e3nn calc + scripts/flash_proto/flashtp_adapter.install (FP64-CG FlashTP from src/flashTP-fp64cg) [as perf2/bench4]
  hybrid : scripts/hybrid/hybrid_backend.build_calc (FlashTP conv + cueq SC)                                [as perf2/bench4]
BACKEND_SET=v2: see build_v2 (filled from results/flash_v2/READY recipe)."""
import os, sys, time
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]
MODEL = f"{R}/models/MACE-OFF23_medium.model"
BACKEND_SET = os.environ.get("BACKEND_SET", "v1")
if BACKEND_SET == "v2":   # READY: import hybrid_v2 before anything else imports flashTP_e3nn
    sys.path.insert(0, f"{R}/scripts/flash_v2"); import hybrid_v2  # noqa
import torch
sync = torch.cuda.synchronize


def tsec(fn):
    sync(); t0 = time.perf_counter(); r = fn(); sync(); return r, (time.perf_counter() - t0) * 1e3


def _mace_calc(**kw):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64", compute_stress=False, **kw)


def _flash_imports():
    sys.path.insert(0, f"{R}/src/flashTP-fp64cg")
    import flashTP_e3nn
    assert flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-fp64cg/"), flashTP_e3nn.__file__
    sys.path.insert(0, f"{R}/scripts/flash_proto")
    import flashtp_adapter as FA
    return flashTP_e3nn, FA


def build_v1(be):
    info = {}
    if be == "e3nn":
        calc, info["construct_ms"] = tsec(lambda: _mace_calc())
    elif be == "cueq":
        calc, info["construct_ms"] = tsec(lambda: _mace_calc(enable_cueq=True))
    elif be == "flash64":
        ftp, FA = _flash_imports()
        def mk():
            c = _mace_calc(); FA.install(c.models[0]); return c
        calc, info["construct_ms"] = tsec(mk)
        assert all(FA.is_installed(calc.models[0]))
        info["flashTP_e3nn"] = ftp.__file__
    elif be == "hybrid":
        _flash_imports(); sys.path.insert(0, f"{R}/scripts/hybrid")
        import hybrid_backend
        (calc, _), info["construct_ms"] = tsec(lambda: hybrid_backend.build_calc(MODEL, "cuda"))
        info["hybrid_backend_file"] = hybrid_backend.__file__
    else:
        raise KeyError(be)
    return calc, info


def build_v2(be):
    """v2 (scripts/flash_v2, cgpatch FlashTP). Names kept as v1: flash64 = conv=flashtp,sc=e3nn ; hybrid = conv=flashtp,sc=cueq."""
    info = {}
    if be in ("e3nn", "cueq"):
        return build_v1(be)
    sys.path.insert(0, f"{R}/scripts/flash_v2")
    import hybrid_v2
    sc = {"flash64": "e3nn", "hybrid": "cueq"}[be]
    ci = os.environ.get("CHECK_INDICES", "0") == "1"
    (calc, hinfo), info["construct_ms"] = tsec(lambda: hybrid_v2.build_calc(MODEL, "cuda", conv="flashtp", sc=sc, dtype=torch.float64, check_indices=ci))
    info["check_indices"] = ci
    info["v2_info"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)[:2000]) for k, v in hinfo.items()}
    info["hybrid_v2_file"] = hybrid_v2.__file__
    return calc, info


def build(be):
    calc, info = (build_v1 if BACKEND_SET == "v1" else build_v2)(be)
    m = calc.models[0]
    info["backend_set"] = BACKEND_SET
    info["param_dtype"] = str(next(m.parameters()).dtype)
    info["module_types"] = sorted({type(x).__module__ + "." + type(x).__name__ for x in m.modules()
                                   if any(s in type(x).__module__ for s in ("cuequivariance", "flash", "hybrid"))})
    return calc, info
