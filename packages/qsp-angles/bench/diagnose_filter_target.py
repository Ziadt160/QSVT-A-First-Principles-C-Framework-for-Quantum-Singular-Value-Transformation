"""Why does the Newton core fail on the eigenstate-filter target, and does the
NLFT path succeed where it fails?

Two properties distinguish the filter from the 1/x targets that solve to machine
precision: it is EVEN (all validated targets were odd) and, at small gap, it is a
near-delta SPIKE. This crosses the two factors:

    A  odd,  smooth   -- near-minimax 1/x            (known good)
    B  even, smooth   -- 0.8 * T_d                   (tests parity alone)
    C  odd,  peaked   -- x * Lin-Tong, renormalised  (tests shape alone)
    D  even, peaked   -- Lin-Tong filter             (the real target)

and runs each through both solvers this project ships.

    python diagnose_filter_target.py
"""
import os
import subprocess
import sys

import numpy as np
from numpy.polynomial import chebyshev as Cheb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "applications", "common"))
sys.path.insert(0, os.path.join(ROOT, "applications", "ground_state"))

import qsp_bridge as Q  # noqa: E402
from eigenstate_filter import lin_tong_filter  # noqa: E402

NLFT_EXE = "/tmp/bench_nlft_bin"
DEG = 60
DELTA = 0.05


def norm_coeffs(c, degree, shave):
    xf = np.cos(np.linspace(0.0, np.pi, 8 * degree + 2))
    sup = float(np.max(np.abs(Cheb.chebval(xf, c))))
    return c / sup * shave


def fit(ys, xs, degree, parity, shave):
    c = Cheb.chebfit(xs, ys, degree)
    c[(1 - parity)::2] = 0.0  # keep only the requested parity
    return norm_coeffs(c, degree, shave)


def targets(shave):
    xs = np.cos(np.linspace(0.0, np.pi, 4 * DEG + 2))
    out = {}

    # A: odd, smooth -- the workload the solver is validated on.
    co, _c, _r = Q.approx_inverse(10.0, DEG - 1 if DEG % 2 == 0 else DEG)
    out["A odd  smooth"] = norm_coeffs(np.asarray(co), len(co) - 1, shave)

    # B: even, smooth -- isolates parity.
    cb = np.zeros(DEG + 1)
    cb[DEG] = 1.0
    out["B even smooth"] = norm_coeffs(cb, DEG, shave)

    # C: odd, peaked -- isolates shape.
    peak = lin_tong_filter(xs, (DEG - 1) // 2 * 1, DELTA)
    out["C odd  peaked"] = fit(xs * peak, xs, DEG - 1, 1, shave)

    # D: even, peaked -- the actual eigenstate filter.
    out["D even peaked"] = fit(lin_tong_filter(xs, DEG // 2, DELTA), xs, DEG, 0, shave)
    return out


def run_nlft(coeffs):
    if not os.path.exists(NLFT_EXE):
        return "no binary"
    r = subprocess.run([NLFT_EXE] + [repr(float(x)) for x in coeffs],
                       capture_output=True, text=True)
    p = r.stdout.split()
    if r.returncode == 0 and p and p[0] == "OK":
        return f"ok {float(p[1]):.2f}s"
    return "FAILED"


def main():
    for shave in (0.999, 0.95):
        print(f"\n=== shave {shave} (eta = {1-shave:.0e}), degree ~{DEG} ===")
        print(f"{'target':>16} {'parity':>7} | {'Newton resid':>13} {'verdict':>9} "
              f"| {'NLFT':>12}")
        for name, c in targets(shave).items():
            deg = len(c) - 1
            par = "even" if deg % 2 == 0 else "odd"
            try:
                _ph, resid = Q.solve_phases(list(c))
                ok = "PASS" if resid < 1e-9 else "FAIL"
                rs = f"{resid:.2e}"
            except Exception as e:
                rs, ok = f"{type(e).__name__}", "FAIL"
            print(f"{name:>16} {par:>7} | {rs:>13} {ok:>9} | {run_nlft(c):>12}",
                  flush=True)
    print("\nDIAGNOSE_DONE")


if __name__ == "__main__":
    main()
