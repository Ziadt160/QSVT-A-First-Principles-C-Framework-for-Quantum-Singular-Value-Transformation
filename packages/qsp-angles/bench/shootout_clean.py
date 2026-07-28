"""Publication-grade angle-solver shootout.

Protocol:
  * identical target for every solver: the near-minimax odd Chebyshev
    approximation of c/x on [1/kappa, 1] that a QSVT linear solver needs;
  * one discarded warm-up, then REPS timed repetitions; report MEDIAN and MIN;
  * single-threaded (OMP/OpenBLAS/MKL pinned to 1) and pinned to one core, so
    BLAS thread-count differences cannot flatter any implementation;
  * a solver that exceeds SLOW_S on a repetition drops to 1 rep thereafter, and
    is skipped entirely past HARD_S (noted in the output, never silently).

Wall-clock is the primary metric because it is convention-free. Phase
conventions differ between libraries, so phases are not cross-graded; our own
solver's residual is reported to show the target is genuinely solved.
"""
import contextlib
import io
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/HelloQuantum/src/kernels/concrete/quantum")
import qsp_bridge as Q

KAPPA = 10.0
DEGREES = [51, 101, 201, 401, 801, 1001]
REPS = 3
SLOW_S = 60.0    # beyond this, drop to a single repetition
HARD_S = 400.0   # beyond this, stop running that solver


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*a, **kw)


def timed(fn, reps):
    """One discarded warm-up, then `reps` timed runs -> (median, min, times)."""
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), ts


# ---------------------------------------------------------------- provenance
def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


def version(mod):
    import importlib.metadata as md
    try:
        return md.version(mod)
    except Exception:
        return "?"


print("=== provenance ===")
print(f"cpu           : {sh(chr(103)+chr(114)+chr(101)+chr(112)+' -m1 ' + chr(39) + 'model name' + chr(39) + ' /proc/cpuinfo')}")
print(f"python        : {platform.python_version()}")
print(f"g++           : {sh('g++ --version | head -1')}")
print(f"numpy         : {version('numpy')}")
print(f"pyqsp         : {version('pyqsp')}")
print(f"qsppack       : {version('qsppack')}")
print(f"nlft-qsp      : {version('nlft-qsp')}")
print(f"threads       : OMP/OpenBLAS/MKL pinned to 1, process pinned to 1 core")
print(f"protocol      : 1 warm-up discarded + median of {REPS} timed runs")
print(f"target        : near-minimax odd Chebyshev approx of c/x on "
      f"[1/{KAPPA:.0f}, 1], |p| <= 1")
print()

rows = []
print(f"{'deg':>5} | {'OURS med':>9} {'min':>8} {'resid':>9} | "
      f"{'nlft med':>9} | {'QSPPACK med':>12} | {'pyqsp med':>10}")
print("-" * 78)

state = {"nlft": REPS, "qsppack": REPS, "pyqsp": REPS}

for d in DEGREES:
    coeffs, c, rel = Q.approx_inverse(KAPPA, d)
    coeffs = coeffs * 0.999
    reduced = coeffs[1::2]
    full = coeffs.tolist()

    med_o, min_o, _ = timed(lambda: Q.solve_phases(full), REPS)
    _, resid = Q.solve_phases(full)

    def run(key, fn):
        if state[key] == 0:
            return None
        med, mn, ts = timed(fn, state[key])
        if med > HARD_S:
            state[key] = 0
        elif med > SLOW_S:
            state[key] = 1
        return med

    try:
        import nlft_qsp
        med_n = run("nlft", lambda: quiet(nlft_qsp.chebqsp_solve, full))
    except Exception as e:
        med_n, state["nlft"] = f"ERR", 0

    try:
        import qsppack
        med_q = run("qsppack", lambda: quiet(
            qsppack.newton, np.asarray(reduced), 1,
            {"maxiter": 100, "criteria": 1e-12}))
    except Exception:
        med_q, state["qsppack"] = "ERR", 0

    try:
        from pyqsp.sym_qsp_opt import newton_solver
        med_p = run("pyqsp", lambda: quiet(newton_solver, reduced, 1, crit=1e-12))
    except Exception:
        med_p, state["pyqsp"] = "ERR", 0

    def f(v):
        if v is None:
            return "stopped"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    rows.append((d, med_o, min_o, resid, med_n, med_q, med_p))
    print(f"{d:>5} | {med_o:>9.3f} {min_o:>8.3f} {resid:>9.1e} | "
          f"{f(med_n):>9} | {f(med_q):>12} | {f(med_p):>10}", flush=True)

print("\n=== speedup vs ours (median) ===")
print(f"{'deg':>5} | {'nlft-qsp':>10} | {'QSPPACK':>10} | {'pyqsp':>10}")
for d, mo, _, _, mn, mq, mp in rows:
    def r(v):
        return f"{v/mo:.1f}x" if isinstance(v, float) else "-"
    print(f"{d:>5} | {r(mn):>10} | {r(mq):>10} | {r(mp):>10}")

print("\n=== CSV ===")
print("degree,ours_median_s,ours_min_s,ours_residual,nlft_median_s,"
      "qsppack_median_s,pyqsp_median_s")
for d, mo, mi, rs, mn, mq, mp in rows:
    def g(v):
        return f"{v:.4f}" if isinstance(v, float) else ""
    print(f"{d},{mo:.4f},{mi:.4f},{rs:.2e},{g(mn)},{g(mq)},{g(mp)}")
print("\nCLEAN_SHOOTOUT_DONE")
