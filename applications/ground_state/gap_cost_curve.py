"""What QSVT ground-state preparation actually COSTS as the spectral gap closes.

The required eigenstate-filter degree scales as ~1/gap, and the gap closes in the
physically interesting regimes -- here the Fermi-Hubbard model driven into the
Mott limit by increasing U/t. So this sweep walks the correlation crossover and
measures, with REAL angles rather than asymptotic bounds:

    U/t  ->  gap  ->  required filter degree  ->  angles  ->  query cost

Why this has not been tabulated before: it needs a few hundred angle solves at
degrees in the thousands. pyqsp/QSPPACK take ~1-70 s per solve (hours for a
sweep); PennyLane cannot do it at all, because its monomial-basis
`poly_to_angles` overflows double precision past degree ~79. Here it is minutes.

The honest framing: this is a RESOURCE ESTIMATE, not a quantum-advantage claim.
The deepest QSP ever executed on hardware is ~360 layers; these degrees are far
beyond any current device. The question answered is "what would this cost?".

*** WHY SAFETY = 0.95 AND NOT 0.999 ***
At a 0.999 shave this whole sweep FAILS: residuals come back 0.7 to 2.0 instead
of ~1e-14, with both a Gaussian fit and the exact closed-form Lin-Tong filter.
It initially looked like a capability gap in the solver -- the filter is EVEN
(every previously validated target was odd) and, at small gap, a near-delta
SPIKE, so either could plausibly have been the cause.

Crossing the two factors (bench/diagnose_filter_target.py) showed it is NEITHER.
An even smooth target and an odd peaked target both solve fine at 0.999; what
fails is any target that is under-resolved for its own parameters combined with
a shave that leaves almost no headroom. Loosening the shave to 0.95 makes every
point here solve to machine precision -- and the sweep runs 4.8x FASTER (70.8 s
vs 338.5 s), because a converging Newton solve is cheap while a failing one
burns its full iteration budget.

The cost of the looser shave is a slightly smaller subnormalization, i.e. a
slightly lower post-selection success probability. That is the correct trade:
angles that exist beat a marginally better amplitude that cannot be realised.

    python gap_cost_curve.py
"""

import math
import os
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as Cheb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import qsp_bridge as Q  # noqa: E402
from fermi_hubbard import hubbard_terms  # noqa: E402
from eigenstate_filter import build_H, pauli_string, lin_tong_filter  # noqa: E402

EPS = 1e-3          # filter suppression target
SAFETY = 0.95       # keep |p| well inside the QSP boundary: at 0.999 these filter
                    # targets do not converge (see diagnose_filter_target.py)


def hubbard_spectrum(t, U):
    """Return (alpha, E0, gap) for the 4-qubit Hubbard Hamiltonian."""
    terms = hubbard_terms(t, U)
    H = build_H(4, terms)
    alpha = sum(abs(c) for c, _p in terms)
    w = np.linalg.eigvalsh(H)
    return alpha, float(w[0]), float(w[1] - w[0])


def lin_tong_degree(delta, eps=EPS):
    """Even degree 2l with suppression <= 2 exp(-2 l delta) outside the gap."""
    l = math.ceil(math.log(2.0 / eps) / (2.0 * delta))
    return 2 * l


def filter_coeffs(degree, delta):
    """Chebyshev coefficients of the CLOSED-FORM Lin-Tong filter, |p| <= 1.

    Uses the exact construction T_l(y(x)) / T_l(y(0)) rather than fitting a
    Gaussian: the closed form IS a degree-2l polynomial, so the Chebyshev fit is
    essentially exact and well conditioned. Fitting a near-delta Gaussian at the
    same degree instead produces a wildly oscillating polynomial which, once
    normalised, is a valid but brutally hard QSP target -- measured residuals of
    7e-4 to 1.0 rather than machine precision.

    Nodes are Chebyshev-clustered; a uniform grid is ill-conditioned at high
    degree and silently yields |p| > 1, an invalid QSP target.
    """
    l = degree // 2
    m = 4 * degree + 2
    xs = np.cos(np.linspace(0.0, np.pi, m))
    ys = lin_tong_filter(xs, l, delta)
    c = Cheb.chebfit(xs, ys, degree)
    c[1::2] = 0.0  # even parity, exactly
    xf = np.cos(np.linspace(0.0, np.pi, 8 * degree + 2))
    sup = float(np.max(np.abs(Cheb.chebval(xf, c))))
    return c / sup * SAFETY


def main():
    t = 1.0
    us = [1.0, 2.0, 4.0, 8.0, 16.0]

    print(f"Fermi-Hubbard (4 qubits), t = {t}. Driving U/t into the Mott limit.\n")
    print(f"{'U/t':>6} {'gap':>9} {'alpha':>8} {'gap/alpha':>10} {'degree':>8} "
          f"{'angle_s':>9} {'resid':>10} {'queries':>10}")
    rows = []
    total_t = 0.0
    for U in us:
        alpha, _e0, gap = hubbard_spectrum(t, U)
        delta = gap / alpha                      # gap of the normalised H
        degree = lin_tong_degree(delta)
        if degree > 6000:
            print(f"{U/t:>6.0f} {gap:>9.4f} {alpha:>8.2f} {delta:>10.4f} "
                  f"{degree:>8} {'(skipped: degree > 6000)':>32}")
            rows.append((U, gap, alpha, delta, degree, float("nan"),
                         float("nan"), float("nan")))
            continue

        c = filter_coeffs(degree, delta)
        t0 = time.perf_counter()
        _ph, resid = Q.solve_phases(c.tolist())
        dt = time.perf_counter() - t0
        total_t += dt

        # Queries per accepted sample: degree queries per application, and the
        # filter's amplitude on the target state sets the acceptance rate.
        queries = float(degree)
        rows.append((U, gap, alpha, delta, degree, dt, resid, queries))
        print(f"{U/t:>6.0f} {gap:>9.4f} {alpha:>8.2f} {delta:>10.4f} {degree:>8} "
              f"{dt:>9.3f} {resid:>10.1e} {queries:>10.0f}", flush=True)

    print(f"\ntotal angle-solve wall-clock: {total_t:.1f} s for "
          f"{sum(1 for r in rows if not math.isnan(r[5]))} solves")

    # Empirical scaling of degree against the normalised gap.
    good = [r for r in rows if not math.isnan(r[5])]
    if len(good) >= 2:
        ld = np.log([r[4] for r in good])
        lg = np.log([r[3] for r in good])
        slope = float(np.polyfit(lg, ld, 1)[0])
        print(f"empirical scaling: degree ~ (gap/alpha)^{slope:.2f}  "
              f"(Lin-Tong predicts ^-1.00)")

    csv = os.path.join(HERE, "gap_cost_curve.csv")
    with open(csv, "w") as fh:
        fh.write("U_over_t,gap,alpha,gap_normalised,degree,angle_time_s,"
                 "angle_residual,queries_per_application\n")
        for r in rows:
            fh.write(",".join("" if isinstance(v, float) and math.isnan(v)
                              else (f"{v:.6g}" if isinstance(v, float) else str(v))
                              for v in r) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
