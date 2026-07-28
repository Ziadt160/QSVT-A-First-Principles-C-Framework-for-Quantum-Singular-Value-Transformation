"""Does the target compiler handle the cases that actually broke things?

Every failure encountered while building this project was a target-construction
failure, never a solver failure. This exercises the compiler against each of
them, plus the cases that must still be REFUSED (with a useful reason rather
than a diverged solve).

    python test_compiler_cases.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "applications", "common"))
sys.path.insert(0, os.path.join(ROOT, "applications", "ground_state"))

# Load compile.py directly: the package __init__ pulls in the pybind `_core`
# extension, which need not be built to use the compiler.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "qsp_compile_mod",
    os.path.join(ROOT, "packages", "qsp-angles", "qsp_angles", "compile.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod  # dataclass resolves annotations via sys.modules
_spec.loader.exec_module(_mod)
qsp_compile = _mod.qsp_compile

from eigenstate_filter import lin_tong_filter  # noqa: E402


def lin_tong(delta):
    l = max(1, math.ceil(math.log(2.0 / 1e-3) / (2.0 * delta)))
    return lambda x: float(lin_tong_filter(np.array([x]), l, delta)[0])


CASES = [
    # (name, f, domain, eps, parity, must_succeed, what it previously broke)
    ("1/x  kappa=10",   lambda x: 1.0 / x, (0.1, 1.0), 1e-3, "odd", True,
     "the standard QSVT inversion target"),
    ("1/x  kappa=100",  lambda x: 1.0 / x, (0.01, 1.0), 1e-3, "odd", True,
     "uniform-grid fits silently gave |p|>1 here -- the 'degree 600 wall'"),
    ("erf(x/0.1)",      lambda x: math.erf(x / 0.1), (0.05, 1.0), 1e-2, "odd", True,
     "tends to +-1, so truncation overshoots |p|=1: no phase sequence exists"),
    ("Lin-Tong d=0.05", lin_tong(0.05), (0.0, 1.0), 1e-2, "even", True,
     "eigenstate filter: failed at a tight shave, fine once loosened"),
    ("Lin-Tong d=0.02", lin_tong(0.02), (0.0, 1.0), 1e-2, "even", True,
     "sharper filter = higher degree"),
    ("x + x^2 (mixed)", lambda x: 0.4 * x + 0.4 * x * x, (0.1, 1.0), 1e-3, None, False,
     "no definite parity: must be REFUSED with an explanation"),
    ("1/x  kappa=1e4",  lambda x: 1.0 / x, (1e-4, 1.0), 1e-6, "odd", False,
     "genuinely out of reach: must say so, not diverge"),
]


def main():
    ok_all = True
    print(f"{'case':>18} {'exp':>5} {'got':>5} {'deg':>6} {'resid':>10} "
          f"{'subnorm':>10} {'shave':>6}")
    print("-" * 72)
    for name, f, dom, eps, par, must, _why in CASES:
        r = qsp_compile(f, domain=dom, eps=eps, parity=par, max_degree=2001)
        got = "ok" if r.ok else "no"
        exp = "ok" if must else "no"
        good = r.ok == must
        ok_all &= good
        if r.ok:
            print(f"{name:>18} {exp:>5} {got:>5} {r.degree:>6} "
                  f"{r.angle_residual:>10.1e} {r.subnorm:>10.3e} {r.shave:>6.3f}"
                  f"  {'PASS' if good else 'FAIL'}")
        else:
            print(f"{name:>18} {exp:>5} {got:>5} {'-':>6} {'-':>10} {'-':>10} "
                  f"{'-':>6}  {'PASS' if good else 'FAIL'}")
            print(f"{'':>18} reason: {r.error.splitlines()[0][:96]}")

    print("\n--- diagnostics the compiler surfaces (sample) ---")
    r = qsp_compile(lin_tong(0.05), domain=(0.0, 1.0), eps=1e-2, parity="even",
                    max_degree=2001)
    for d in r.diagnostics[:6]:
        print(f"  {d}")

    print("\n" + ("ALL PASS" if ok_all else "FAILURES PRESENT"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
