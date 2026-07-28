"""Consolidated benchmark: every open-source QSP angle solver we could find,
on identical near-minimax 1/x QSVT targets, across the two axes that matter
(polynomial degree, and eta = 1 - sup|p|).

Solvers
  ours-newton  C++ symmetric-QSP Newton (sym_qsp)      this project, default
  ours-nlft    C++ Weiss + Half-Cholesky inverse NLFT  this project
  pyqsp        Python, same Newton algorithm as ours   MIT
  QSPPACK      Python port of the MATLAB package       Newton
  nlft-qsp     Python, Weiss + divide-and-conquer NLFT MIT, mpmath by default

Not benchmarked, because they cannot run this workload at all:
  PennyLane    poly_to_angles takes MONOMIAL coefficients; converting a degree-79
               Chebyshev inversion polynomial overflows double (coeffs ~1e25), so
               a target with true sup-norm 0.999 is rejected as |P| > 1.
  Qiskit       ships no QSP/QSVT angle solver.

    python bench_all_solvers.py
"""
import contextlib
import io
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "applications", "common"))

import qsp_bridge as Q  # noqa: E402

NLFT_EXE = "/tmp/bench_nlft_bin"  # built by bench_nlft_vs_newton.py
CASES = [(10.0, 79), (30.0, 247), (60.0, 505)]
SHAVES = [0.999, 0.95]


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*a, **kw)


def t(fn):
    t0 = time.perf_counter()
    try:
        fn()
        return time.perf_counter() - t0
    except Exception:
        return float("nan")


def main():
    if not os.path.exists(NLFT_EXE):
        print(f"note: {NLFT_EXE} missing; run bench_nlft_vs_newton.py first")

    print("Identical near-minimax 1/x targets. Seconds; lower is better.\n")
    hdr = (f"{'kappa':>6} {'deg':>5} {'eta':>7} | {'ours-newton':>12} "
           f"{'ours-nlft':>10} | {'pyqsp':>9} {'QSPPACK':>9} {'nlft-qsp':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for kap, d in CASES:
        base, _c, _r = Q.approx_inverse(kap, d)
        for shave in SHAVES:
            coeffs = (base * shave).tolist()
            reduced = np.asarray(coeffs)[1::2]
            eta = 1.0 - shave

            t_new = t(lambda: Q.solve_phases(coeffs))

            if os.path.exists(NLFT_EXE):
                r = subprocess.run([NLFT_EXE] + [repr(float(x)) for x in coeffs],
                                   capture_output=True, text=True)
                p = r.stdout.split()
                t_onl = float(p[1]) if r.returncode == 0 and p and p[0] == "OK" else float("nan")
            else:
                t_onl = float("nan")

            try:
                from pyqsp.sym_qsp_opt import newton_solver
                t_py = t(lambda: quiet(newton_solver, reduced, 1, crit=1e-12))
            except Exception:
                t_py = float("nan")

            try:
                import qsppack
                t_qp = t(lambda: quiet(qsppack.newton, reduced, 1,
                                       {"maxiter": 100, "criteria": 1e-12}))
            except Exception:
                t_qp = float("nan")

            try:
                import nlft_qsp
                t_nl = t(lambda: quiet(nlft_qsp.chebqsp_solve, coeffs))
            except Exception:
                t_nl = float("nan")

            def f(x):
                return "  -  " if np.isnan(x) else f"{x:.3f}"

            rows.append((kap, d, eta, t_new, t_onl, t_py, t_qp, t_nl))
            print(f"{kap:>6.0f} {d:>5} {eta:>7.0e} | {f(t_new):>12} {f(t_onl):>10} "
                  f"| {f(t_py):>9} {f(t_qp):>9} {f(t_nl):>9}", flush=True)

    print("\n=== best-of-this-project vs best-external, per row ===")
    print(f"{'kappa':>6} {'deg':>5} {'eta':>7} | {'ours best':>10} "
          f"{'external best':>14} {'advantage':>10}")
    for kap, d, eta, tn, to, tp, tq, tl in rows:
        ours = np.nanmin([tn, to])
        ext = np.nanmin([tp, tq, tl])
        adv = ext / ours if ours > 0 else float("nan")
        print(f"{kap:>6.0f} {d:>5} {eta:>7.0e} | {ours:>10.3f} {ext:>14.3f} "
              f"{adv:>9.1f}x")

    print("\nALL_SOLVERS_DONE")


if __name__ == "__main__":
    main()
