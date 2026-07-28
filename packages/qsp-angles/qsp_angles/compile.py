"""QSP target compiler: from a function to validated phase angles.

Every other QSP package starts one step too late. pyqsp, QSPPACK and nlft-qsp all
take *coefficients* and trust that you produced a valid target; PennyLane's
monomial-basis API actively destroys one. But in practice the failures are almost
never in the phase-factor solver -- they are in the polynomial handed to it:

  * a uniform-grid Chebyshev fit silently yields |p| > 1 above degree ~600, an
    unrealisable target that makes a correct solver diverge from iteration 0;
  * a target approaching +-1 asymptotically (erf, sign approximations) overshoots
    |p| = 1 under truncation and has no phase sequence at all;
  * a degree chosen by hand rather than from (kappa, eps) is under-resolved, and
    an under-resolved target is a non-convergence, not a slightly worse answer;
  * a shave that leaves no headroom turns an otherwise fine target unsolvable.

`qsp_compile` owns that step. Give it a callable, a domain and an accuracy; it
returns angles that are known-good, or an explanation naming the actual problem
and what to change.

    from qsp_angles.compile import qsp_compile
    r = qsp_compile(lambda x: 1/x, domain=(0.1, 1.0), eps=1e-3, parity="odd")
    if r.ok:
        use(r.angles)          # residual r.angle_residual, subnormalization r.subnorm
    else:
        print(r.error)         # e.g. "under-resolved: ... raise max_degree above 4001"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.polynomial import chebyshev as Cheb

__all__ = ["CompileResult", "qsp_compile"]

# Shave ladder: start tight (best subnormalization, so best success probability)
# and loosen only as needed. A looser shave costs amplitude but is often the
# difference between angles that exist and angles that do not.
_SHAVES = (0.999, 0.99, 0.95, 0.9)
_RESID_OK = 1e-9


@dataclass
class CompileResult:
    """Outcome of a compile. Check `ok` first; on failure read `error`."""

    ok: bool
    angles: Optional[np.ndarray] = None
    coeffs: Optional[np.ndarray] = None
    degree: Optional[int] = None
    parity: Optional[str] = None
    subnorm: float = 0.0          # c: p(x) ~ c*f(x); sets success probability ~c^2
    sup_norm: float = 0.0         # achieved max|p| on [-1, 1]; must be <= 1
    poly_rel_err: float = float("nan")   # worst relative error vs f on the domain
    angle_residual: float = float("nan")
    shave: float = float("nan")
    solver: str = "newton"
    diagnostics: list = field(default_factory=list)
    error: Optional[str] = None

    def __str__(self) -> str:
        if not self.ok:
            return f"CompileResult(FAILED: {self.error})"
        return (f"CompileResult(ok, degree={self.degree}, parity={self.parity}, "
                f"residual={self.angle_residual:.1e}, subnorm={self.subnorm:.3e}, "
                f"shave={self.shave})")


def _default_solver():
    """Lazily resolve a phase solver so the module imports without a built .so."""
    try:
        from ._core import sym_qsp_poly_to_angles  # type: ignore

        def solve(coeffs):
            out = sym_qsp_poly_to_angles(list(coeffs))
            return np.asarray(out[0]), float(out[1])

        return solve
    except Exception:
        pass
    import qsp_bridge as bridge  # type: ignore

    return bridge.solve_phases


def _cheb_nodes(m: int) -> np.ndarray:
    """Endpoint-clustered nodes. A uniform grid is ill-conditioned at high degree
    and steps over the endpoint oscillation, which is how |p| > 1 goes unnoticed."""
    return np.cos(np.linspace(0.0, np.pi, m))


def _detect_parity(f: Callable[[float], float], lo: float, hi: float) -> str:
    xs = lo + (hi - lo) * (np.linspace(0.05, 0.95, 17))
    fp = np.array([f(x) for x in xs], dtype=float)
    fm = np.array([f(-x) for x in xs], dtype=float)
    scale = max(1e-300, float(np.max(np.abs(fp))))
    if float(np.max(np.abs(fm - fp))) / scale < 1e-8:
        return "even"
    if float(np.max(np.abs(fm + fp))) / scale < 1e-8:
        return "odd"
    return "indefinite"


def _fit(f, lo, hi, degree, parity):
    """Least-squares fit of f on |x| in [lo, hi] in the given-parity Chebyshev
    basis, normalised to unit sup-norm on [-1, 1].

    Returns (coeffs, c, rel_err, raw_sup) where c is the subnormalization.
    """
    m = max(2000, 4 * degree + 2)
    u = _cheb_nodes(m)
    x = lo + (hi - lo) * (u + 1.0) / 2.0          # nodes mapped onto [lo, hi]
    y = np.array([f(xi) for xi in x], dtype=float)

    start = 1 if parity == "odd" else 0
    idx = list(range(start, degree + 1, 2))
    V = Cheb.chebvander(x, degree)[:, idx]
    a, *_ = np.linalg.lstsq(V, y, rcond=None)

    coeffs = np.zeros(degree + 1)
    coeffs[idx] = a

    xf = _cheb_nodes(8 * degree + 2)
    sup = float(np.max(np.abs(Cheb.chebval(xf, coeffs))))
    if sup <= 0.0 or not np.isfinite(sup):
        return coeffs, 0.0, float("inf"), sup
    coeffs = coeffs / sup
    c = 1.0 / sup

    xd = _cheb_nodes(4 * degree + 2)
    xd = xd[np.abs(xd) >= lo]
    ref = np.array([f(xi) for xi in xd], dtype=float) * c
    got = Cheb.chebval(xd, coeffs)
    denom = np.maximum(np.abs(ref), 1e-300)
    rel = float(np.max(np.abs(got - ref) / denom))
    return coeffs, c, rel, sup


def _min_degree(f, lo, hi, eps, parity, max_degree):
    """Smallest degree of the right parity meeting `eps`, by geometric bracket
    then bisection -- O(log) fits rather than O(max_degree)."""
    def rel_at(d):
        return _fit(f, lo, hi, d, parity)[2]

    step = 2
    d = 1 if parity == "odd" else 2
    if rel_at(d) <= eps:
        return d
    prev = d
    while d <= max_degree:
        prev, d = d, d * 2
        if d > max_degree:
            break
        if rel_at(d) <= eps:
            hi_d, lo_d = d, prev
            while hi_d - lo_d > step:
                mid = (lo_d + hi_d) // 2
                if (mid % 2) != (1 if parity == "odd" else 0):
                    mid += 1
                if mid >= hi_d:
                    break
                if rel_at(mid) <= eps:
                    hi_d = mid
                else:
                    lo_d = mid
            return hi_d
    return None


def qsp_compile(
    f: Callable[[float], float],
    domain: Sequence[float] = (0.0, 1.0),
    eps: float = 1e-3,
    parity: Optional[str] = None,
    max_degree: int = 4001,
    solve_phases: Optional[Callable] = None,
) -> CompileResult:
    """Compile a target function into QSP phase angles.

    Args:
        f: the target, evaluated on |x| in [lo, hi]. It is realised up to a
           subnormalization: the angles implement ``c * f`` with ``|c*f| <= 1``.
        domain: (lo, hi) with 0 <= lo < hi <= 1. For QSVT inversion at condition
           number kappa this is ``(1/kappa, 1.0)``.
        eps: target worst-case *relative* error of the polynomial against f.
        parity: "odd", "even", or None to detect. QSP requires definite parity.
        max_degree: refuse to search past this.
        solve_phases: injected solver (coeffs -> (angles, residual)).

    Returns:
        CompileResult. On failure `error` names the actual cause.
    """
    diag: list = []
    lo, hi = float(domain[0]), float(domain[1])
    if not (0.0 <= lo < hi <= 1.0):
        return CompileResult(False, error=(
            f"domain must satisfy 0 <= lo < hi <= 1, got ({lo}, {hi}). QSP acts on "
            f"the singular values of a block-encoded operator, which live in [0, 1]."))

    if parity is None:
        parity = _detect_parity(f, lo, hi)
        diag.append(f"detected parity: {parity}")
    if parity == "indefinite":
        return CompileResult(False, parity=parity, diagnostics=diag, error=(
            "target has no definite parity. QSP realises only even or odd "
            "polynomials; split f into f_even + f_odd and compile each, or apply "
            "the standard linear-combination trick with an extra ancilla."))

    degree = _min_degree(f, lo, hi, eps, parity, max_degree)
    if degree is None:
        _, _, rel, _ = _fit(f, lo, hi, max_degree, parity)
        return CompileResult(False, parity=parity, diagnostics=diag, error=(
            f"under-resolved: at max_degree={max_degree} the best relative error is "
            f"{rel:.2e}, short of eps={eps:.1e}. Raise max_degree, loosen eps, or "
            f"reduce the difficulty of the domain (lo={lo:g} is the binding "
            f"parameter; cost grows ~1/lo)."))
    diag.append(f"minimal degree for eps={eps:.1e}: {degree}")

    if solve_phases is None:
        solve_phases = _default_solver()

    # Escalate: loosen the shave, then bump the degree. Both are cheap relative
    # to shipping angles that do not exist.
    last_resid = float("nan")
    for bump in (0, 6, 12, 24):
        d = degree + (bump if (degree + bump) % 2 == degree % 2 else bump + 1)
        if d > max_degree:
            break
        coeffs, c, rel, raw_sup = _fit(f, lo, hi, d, parity)
        if not np.isfinite(raw_sup):
            continue
        for shave in _SHAVES:
            trial = coeffs * shave
            try:
                angles, resid = solve_phases(list(trial))
            except Exception as exc:  # solver refused outright
                last_resid = float("nan")
                diag.append(f"degree {d}, shave {shave}: solver raised {type(exc).__name__}")
                continue
            last_resid = float(resid)
            if np.isfinite(resid) and resid < _RESID_OK:
                if shave != _SHAVES[0]:
                    diag.append(
                        f"shave loosened to {shave} (tighter shaves left too little "
                        f"headroom); subnormalization reduced to {c*shave:.3e}")
                if bump:
                    diag.append(f"degree raised {degree} -> {d} for a clean solve")
                sup = float(np.max(np.abs(Cheb.chebval(_cheb_nodes(8 * d + 2), trial))))
                return CompileResult(
                    ok=True, angles=np.asarray(angles), coeffs=trial, degree=d,
                    parity=parity, subnorm=c * shave, sup_norm=sup,
                    poly_rel_err=rel, angle_residual=float(resid), shave=shave,
                    solver="newton", diagnostics=diag)
            diag.append(f"degree {d}, shave {shave}: residual {resid:.2e}")

    return CompileResult(False, parity=parity, degree=degree, diagnostics=diag,
                         angle_residual=last_resid, error=(
        f"no valid phase sequence found. Best residual {last_resid:.2e} after "
        f"trying degrees {degree}..{degree+24} and shaves {_SHAVES}. This is "
        f"almost always the target, not the solver: check that f is genuinely "
        f"realisable (|c*f| <= 1 on all of [-1, 1], not just on the domain), and "
        f"that lo={lo:g} is not so small that the required degree exceeds "
        f"max_degree={max_degree}."))
