"""Same as parity_model.py but imports flashTP_e3nn from the experimental worktree src/flashTP-fp64cg (branch exp/fp64-cg)."""
import os, sys, runpy
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
R = os.environ["R"]; sys.path.insert(0, f"{R}/src/flashTP-fp64cg")
import flashTP_e3nn; assert flashTP_e3nn.__file__.startswith(f"{R}/src/flashTP-fp64cg"), flashTP_e3nn.__file__
print("flashTP_e3nn from", flashTP_e3nn.__file__)
sys.argv = sys.argv[:]; runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_model.py"), run_name="__main__")
