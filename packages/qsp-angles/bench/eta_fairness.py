"""Fairness check: nlft-qsp documents that its runtime scales with 1/eta, where
eta = 1 - sup|P|. Our shootout shaved targets to 0.999 (eta = 1e-3), which by that
documentation is a near-worst case for it. If nlft-qsp speeds up materially with a
larger shave, our benchmark understated it and the crossover is earlier than
measured.

Our Newton solver's cost should be roughly indifferent to eta.
"""
import contextlib
import io
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/HelloQuantum/src/kernels/concrete/quantum")
import qsp_bridge as Q
import nlft_qsp


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*a, **kw)


DEGREE, KAPPA = 1001, 110.0
print(f"degree={DEGREE}, kappa={KAPPA}; varying the sup-norm shave\n")
print(f"{'shave':>7} {'eta':>8} | {'ours_s':>8} {'our_resid':>11} | {'nlft_s':>8}")
print("-" * 56)

base, c, _ = Q.approx_inverse(KAPPA, DEGREE)
for shave in (0.999, 0.99, 0.95, 0.90):
    coeffs = base * shave
    eta = 1.0 - shave

    t0 = time.perf_counter()
    ph, resid = Q.solve_phases(coeffs.tolist())
    t_ours = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        quiet(nlft_qsp.chebqsp_solve, coeffs.tolist())
        t_nlft = time.perf_counter() - t0
        s_nlft = f"{t_nlft:8.2f}"
    except Exception as e:
        s_nlft = f"ERR:{type(e).__name__}"

    print(f"{shave:>7.3f} {eta:>8.0e} | {t_ours:>8.2f} {resid:>11.1e} | {s_nlft:>8}",
          flush=True)

print("\nETA_DONE")
