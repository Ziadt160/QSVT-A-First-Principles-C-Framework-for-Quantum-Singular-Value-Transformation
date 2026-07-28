"""Find where nlft-qsp's O(d log^2 d) method overtakes our ~O(d^2) Newton core.

Degrees 1501-4001. kappa is scaled with degree (kappa ~ d/9.1) so each target is
in its REALISTIC regime -- a degree-4001 polynomial is needed because the system
is ill-conditioned, not because we over-provisioned a well-conditioned one. (At
kappa=10 a degree-4001 fit is rank-deficient and meaningless: degree 79 suffices.)

Timing protocol: median of 3 (with a discarded warm-up) up to degree 2001; a
single timed run above that, because each run costs minutes. Reduced sampling is
reported per row, never silently applied.
"""
import contextlib
import io
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/HelloQuantum/src/kernels/concrete/quantum")
import qsp_bridge as Q
import nlft_qsp

DEGREES = [1501, 2001, 3001, 4001]


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*a, **kw)


def timed(fn, reps, warmup):
    if warmup:
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print("Crossover sweep: ours (C++ Newton) vs nlft-qsp (inverse nonlinear FFT).")
print("kappa scaled with degree (~d/9.1) so every target is in its real regime.\n")
print(f"{'degree':>7} {'kappa':>7} {'reps':>5} | {'ours':>9} {'resid':>10} | "
      f"{'nlft-qsp':>10} | {'ratio':>9}")
print("-" * 74)

for d in DEGREES:
    kap = round(d / 9.1)
    reps, warm = (3, True) if d <= 2001 else (1, False)

    coeffs, c, rel = Q.approx_inverse(kap, d)
    coeffs = coeffs * 0.999
    full = coeffs.tolist()

    t_ours = timed(lambda: Q.solve_phases(full), reps, warm)
    _, resid = Q.solve_phases(full)

    try:
        t_nlft = timed(lambda: quiet(nlft_qsp.chebqsp_solve, full), reps, warm)
        ratio = f"{t_nlft / t_ours:.2f}x"
    except Exception as e:
        t_nlft, ratio = float("nan"), f"ERR:{type(e).__name__}"

    lead = "ours" if t_nlft > t_ours else "NLFT WINS"
    print(f"{d:>7} {kap:>7} {reps:>5} | {t_ours:>9.2f} {resid:>10.1e} | "
          f"{t_nlft:>10.2f} | {ratio:>9}  <- {lead}", flush=True)

print("\nCROSSOVER_DONE")
