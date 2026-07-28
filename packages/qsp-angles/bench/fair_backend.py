"""Our published crossover compared C++ double against nlft-qsp on its DEFAULT
backend -- which is mpmath arbitrary precision (machine_eps = 1.08e-19), not
double. That is not a like-for-like comparison. Re-time it on the numpy (double)
backend, which is the honest one.
"""
import contextlib
import io
import statistics
import sys
import time

sys.path.insert(0, "/tmp/HelloQuantum/src/kernels/concrete/quantum")
import qsp_bridge as Q
import nlft_qsp
import nlft_qsp.numerics as bd
from nlft_qsp.numerics.backend_numpy import NumpyBackend


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*a, **kw)


def timed(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


DEFAULT_BACKEND = bd.backend  # whatever nlft-qsp ships as the default (mpmath)
print(f"default backend machine_eps = {float(bd.machine_eps()):.3e}")
print(f"{'deg':>5} | {'ours (C++)':>11} | {'nlft mpmath':>12} | {'nlft numpy':>11} "
      f"| {'ratio vs ours':>14}")
print("-" * 70)

for d in (51, 101, 201, 401, 801):
    coeffs, c, _ = Q.approx_inverse(10.0, d)
    coeffs = coeffs * 0.999
    full = coeffs.tolist()

    t_ours = timed(lambda: Q.solve_phases(full))

    bd.set_backend(DEFAULT_BACKEND)
    try:
        t_mp = timed(lambda: quiet(nlft_qsp.chebqsp_solve, full), reps=1)
    except Exception:
        t_mp = float("nan")

    bd.set_backend(NumpyBackend())
    try:
        t_np = timed(lambda: quiet(nlft_qsp.chebqsp_solve, full), reps=3)
        r = f"{t_np / t_ours:.2f}x"
    except Exception as e:
        t_np, r = float("nan"), f"ERR:{type(e).__name__}"

    print(f"{d:>5} | {t_ours:>11.3f} | {t_mp:>12.3f} | {t_np:>11.3f} | {r:>14}",
          flush=True)

print("\nFAIR_DONE")
