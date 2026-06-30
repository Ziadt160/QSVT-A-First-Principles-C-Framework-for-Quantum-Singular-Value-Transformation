"""Timing/accuracy harness for pyqsp's sym_qsp solver, for the SAME scaled
Chebyshev target c*T_d as bench_core.cpp.

Each solver is graded in its OWN convention (pyqsp's phase gauge differs from
this package's canonical Wx product, so cross-evaluating phases is not
apples-to-apples). We therefore report pyqsp's self-reported optimizer residual
(the final "err" from its Newton iteration) -- pyqsp grading itself -- which is
the fair yardstick against the core solver's own grid residual.

Prints CSV: degree,pyqsp_time_ms,pyqsp_residual,nphases

    pip install pyqsp        # numpy/scipy-based, pure Python
    python bench_pyqsp.py
"""

import contextlib
import io
import re
import statistics
import time

import numpy as np
from pyqsp.angle_sequence import QuantumSignalProcessingPhases as QSP

DEGREES = [11, 21, 31, 51, 71, 101]
C = 0.8


def run(d):
    cheb = [0.0] * (d + 1)
    cheb[d] = C
    times, last_err, nph = [], float("nan"), 0
    for _ in range(3):
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            full, _reduced, _parity = QSP(
                np.array(cheb), signal_operator="Wx", method="sym_qsp", chebyshev_basis=True
            )
        times.append((time.perf_counter() - t0) * 1e3)
        nph = len(full)
        errs = re.findall(r"err:\s*([0-9.eE+-]+)", buf.getvalue())
        if errs:
            last_err = float(errs[-1])
    return statistics.median(times), last_err, nph


def main():
    run(11)  # warm up (import/JIT)
    print("degree,pyqsp_time_ms,pyqsp_residual,nphases")
    for d in DEGREES:
        t, err, n = run(d)
        print(f"{d},{t:.1f},{err:.2e},{n}")


if __name__ == "__main__":
    main()
