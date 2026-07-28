"""W1 spike: a classically-validated odd polynomial approximation of 1/x for QSVT
quantum linear systems. NO quantum here -- this decouples approximation error
from QSVT error and measures the degree-vs-kappa scaling the whole Phase-1 claim
rests on.

Target: an odd polynomial p with  p(x) ≈ c/x  on the domain
D = [1/κ, 1] ∪ [−1, −1/κ], scaled so |p| ≤ 1 on [−1, 1] (a valid QSP target).
The QSVT then realises p(A) ≈ c·A⁻¹; the subnormalization c sets the (un-amplified)
success probability ~ (c·‖A⁻¹|b>‖)². p's Chebyshev coefficients feed sym_qsp directly.

Two constructions are compared:
  * least-squares fit in the odd Chebyshev basis on D (near-minimax)  -> ~O(κ·polylog)
  * the Childs-Kothari-Somma closed form (1-(1-x²)^b)/x (baseline)    -> ~O(κ²)

    python oneoverx_approx.py
"""

import math

import numpy as np
from numpy.polynomial import chebyshev as Cheb


def approximate_inverse(kappa: float, degree: int, npts: int = 4000):
    """Odd degree-`degree` Chebyshev fit of c/x on D, normalized to sup-norm 1.

    Fit nodes and the sup-norm grid are CHEBYSHEV-distributed (clustered at the
    ends), not uniform. Above degree ~600 a uniform-grid least-squares fit of the
    steep c/x is ill-conditioned and oscillates to O(10^2) between nodes near
    x=+-1, and a uniform sup grid steps over those spikes -- so the "normalized"
    coefficients silently described a polynomial with true sup-norm >>1, i.e. NOT
    a valid QSP target. Chebyshev nodes keep the Vandermonde well-conditioned and
    the measured sup honest, so |p|<=1 holds through degree >1000.

    Returns (coeffs, c, rel_err) where `coeffs` are full Chebyshev coefficients
    (length degree+1, even entries 0), `c` is the effective subnormalization, and
    `rel_err` is the worst-case relative error on D. (Relative error is invariant
    under jointly scaling p and c, so the sup-norm normalization is free.)
    """
    delta = 1.0 / kappa
    c0 = 1.0 / (2.0 * kappa)  # any c works; rescaled below
    # Chebyshev-distributed fit nodes on [delta, 1], oversampled ~4x, so the
    # degree-`degree` Vandermonde stays well-conditioned.
    m = max(npts, 4 * degree + 2)
    u = np.cos(np.linspace(0.0, np.pi, m))          # Chebyshev abscissae in [-1,1]
    x = delta + (1.0 - delta) * (u + 1.0) / 2.0     # mapped onto [delta, 1]
    y = c0 / x

    odd = list(range(1, degree + 1, 2))
    V = Cheb.chebvander(x, degree)  # columns T_0..T_degree
    a, *_ = np.linalg.lstsq(V[:, odd], y, rcond=None)
    coeffs = np.zeros(degree + 1)
    coeffs[odd] = a

    # Normalize to unit sup-norm on [-1, 1]. |p| peaks either near x=+-1 (a
    # high-degree fit can spike there) or near the gap edge x~+-delta (the 1/x
    # approx's own peak). Measure on a grid dense at BOTH -- Chebyshev abscissae
    # (cluster at +-1) plus a fine linear grid across the gap band -- since a
    # single Chebyshev grid is sparse near 0 and undersamples the gap peak.
    edge = min(1.0, 4.0 * delta)
    xf = np.concatenate([
        np.cos(np.linspace(0.0, np.pi, 8 * degree + 2)),
        np.linspace(-edge, edge, max(20000, 8 * degree)),
    ])
    sup = float(np.max(np.abs(Cheb.chebval(xf, coeffs))))
    coeffs /= sup
    c = c0 / sup

    xd = np.cos(np.linspace(0.0, np.pi, 4 * degree + 2))
    xd = xd[xd >= delta]
    rel = float(np.max(np.abs(Cheb.chebval(xd, coeffs) - c / xd) / np.abs(c / xd)))
    return coeffs, c, rel


def min_degree(kappa: float, eps: float, dmax: int = 800):
    """Smallest odd degree whose fit reaches relative error <= eps on D."""
    for d in range(1, dmax + 1, 2):
        _coeffs, c, rel = approximate_inverse(kappa, d)
        if rel <= eps:
            return d, rel, c
    return None, None, None


def cks_degree(kappa: float, eps: float) -> int:
    """Degree of the Childs-Kothari-Somma (1-(1-x²)^b)/x baseline for rel err eps.

    Relative error on D is bounded by (1-δ²)^b with δ=1/κ, so b ≈ ln(1/eps)/δ²;
    degree = 2b-1 ~ O(κ² log 1/eps).
    """
    delta = 1.0 / kappa
    b = math.ceil(math.log(1.0 / eps) / (-math.log(1.0 - delta * delta)))
    return 2 * b - 1


def _slope_loglog(kappas, degrees):
    """Empirical exponent: fit log(degree) = m log(kappa) + const, return m."""
    lk = np.log(np.array(kappas, float))
    ld = np.log(np.array(degrees, float))
    m, _b = np.polyfit(lk, ld, 1)
    return float(m)


def main():
    kappas = [2, 4, 8, 16, 32, 64]
    for eps in (1e-2, 1e-3):
        print(f"\n=== target relative error eps = {eps:.0e} ===")
        print(f"{'kappa':>6} {'fit_deg':>8} {'fit_relerr':>12} {'c (subnorm)':>12} {'CKS_deg':>8}")
        fit_degs = []
        for k in kappas:
            d, rel, c = min_degree(k, eps)
            cks = cks_degree(k, eps)
            fit_degs.append(d)
            print(f"{k:>6} {d:>8} {rel:>12.2e} {c:>12.3e} {cks:>8}")
        m = _slope_loglog(kappas, fit_degs)
        print(f"  fit degree scales ~ kappa^{m:.2f}  (CKS baseline ~ kappa^2)")

    # Sanity on one representative target (kappa=10, eps=1e-3): confirm |p|<=1.
    coeffs, c, rel = approximate_inverse(10.0, min_degree(10.0, 1e-3)[0])
    xf = np.linspace(-1, 1, 20001)
    print(
        f"\nsanity (kappa=10): degree={len(coeffs)-1}, rel_err={rel:.2e}, "
        f"c={c:.3e}, max|p| on [-1,1]={np.max(np.abs(Cheb.chebval(xf, coeffs))):.6f}"
    )


if __name__ == "__main__":
    main()
