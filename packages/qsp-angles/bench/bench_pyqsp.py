"""pyqsp timing baseline -- times pyqsp's own `newton_solver` directly.

This deliberately times the SAME function the C++ port implements
(`pyqsp.sym_qsp_opt.newton_solver`, the Dong-Lin-Ni-Wang symmetric-QSP Newton
method), on the same scaled-Chebyshev target `0.8*T_d` and the same degrees as
`bench_symqsp_core.cpp`. So the two CSVs line up degree-for-degree and the
comparison is genuinely like-for-like (compiled vs interpreted, same algorithm)
-- NOT pyqsp's high-level `QuantumSignalProcessingPhases`, which wraps extra
work and is not what the C++ core does.

Prints CSV: degree,pyqsp_time_ms,iters

    pip install pyqsp
    python bench_pyqsp.py
"""

import contextlib
import io
import statistics
import time

import numpy as np
from pyqsp.sym_qsp_opt import newton_solver

DEGREES = [11, 21, 51, 101, 201, 501, 1001]
C = 0.8


def run(d, reps):
    n = d // 2 + 1
    parity = d % 2
    coef = np.zeros(n)
    coef[-1] = C  # 0.8 * T_d in the parity-reduced Chebyshev basis
    times, iters = [], 0
    for _ in range(reps):
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            _red, _err, iters, _obj = newton_solver(coef.copy(), parity, crit=1e-12, maxiter=100)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), iters


def main():
    run(11, 1)  # warm up
    print("degree,pyqsp_time_ms,iters")
    for d in DEGREES:
        reps = 3 if d <= 101 else 1  # high degrees are slow and low-variance
        t, iters = run(d, reps)
        print(f"{d},{t:.1f},{iters}")


if __name__ == "__main__":
    main()
