"""qsp-angles: a fast C++ QSP/QSVT phase-factor (angle) solver.

The heavy lifting is a self-contained C++ solver exposed to Python via pybind11.
Two methods are available (select via ``method=``):

* ``"sym_qsp"`` (DEFAULT) -- a symmetric-QSP Newton method (Dong-Lin-Ni-Wang,
  arXiv:2307.12468; the algorithm behind pyqsp's ``sym_qsp``). It is strictly
  faster than the homotopy solver and reaches degree 1000+ at machine precision.
* ``"homotopy"`` -- homotopy continuation + an exact analytic Jacobian
  (Levenberg-Marquardt over Chebyshev nodes), kept as a fallback.

Both compute the phase sequence ``Phi`` such that, in the **Wx convention**,

    U(x, Phi) = e^{i phi_0 Z} prod_{k>=1} [ W(x) e^{i phi_k Z} ],
    W(x) = e^{i arccos(x) X},

realises ``Re<0|U(x)|0> = f(x)`` for a chosen real target ``f`` with
``|f| <= 1`` on ``[-1, 1]`` and definite parity ``deg mod 2``.

Quick start
-----------
>>> import qsp_angles as qa
>>> r = qa.target2angles(lambda x: 0.7 * x, degree=1)
>>> r.converged
True
>>> abs(qa.response(0.3, r.phases) - 0.7 * 0.3) < 1e-9
True

Polynomial coefficients default to the **Chebyshev** basis (the QSP/QSVT field
convention). Pass ``basis="monomial"`` for ascending power-basis coefficients:

>>> import numpy as np
>>> r = qa.poly2angles([0.0, 0.0, 1.0])            # T_2 = 2 x^2 - 1
>>> r = qa.poly2angles([0.0, 0.0, 1.0], basis="monomial")  # x^2

A familiar convenience wrapper
------------------------------
``QuantumSignalProcessingPhases`` is named after the pyqsp entry point for
familiarity, but it is **not** a drop-in replacement -- see its docstring and
the README "Differences from pyqsp" note.

>>> from qsp_angles import QuantumSignalProcessingPhases
>>> phases = QuantumSignalProcessingPhases([0.0, 0.7], signal_operator="Wx")  # 0.7 * x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Union

import numpy as np

from . import _core

__all__ = [
    "AngleResult",
    "target2angles",
    "poly2angles",
    "response",
    "QuantumSignalProcessingPhases",
]

__version__ = "0.1.0"

PolyLike = Union[Callable[[float], float], Sequence[float], np.ndarray, "np.polynomial.Polynomial"]

# Default convergence tolerance on the worst-case grid residual.
_DEFAULT_TOL = 1e-6

# Supported angle-solver methods, mapping the public ``method`` name to its
# ``_core`` entry point. ``"sym_qsp"`` (the symmetric-QSP Newton method,
# Dong-Lin-Ni-Wang arXiv:2307.12468) is the DEFAULT: it is strictly faster than
# the homotopy solver and reaches far higher degree (1000+) at machine
# precision. ``"homotopy"`` (homotopy continuation + analytic Jacobian) is kept
# as a fallback. Both return {phases, residual, converged} in the Wx / Re
# convention, so they are interchangeable at the call site.
_DEFAULT_METHOD = "sym_qsp"
_METHODS = {
    "sym_qsp": "sym_qsp_poly_to_angles",
    "homotopy": "poly_to_angles",
}


@dataclass
class AngleResult:
    """Result of an angle solve.

    Attributes:
        phases: length ``degree + 1`` symmetric phase sequence (Wx convention).
        residual: worst-case ``|Re<0|U|0> - f|`` over a fine grid on [-1, 1]
            (reported by the C++ solver).
        converged: whether ``residual < tol`` (the ``tol`` passed to the solve;
            default ``1e-6``).
    """

    phases: np.ndarray
    residual: float
    converged: bool

    def __array__(self, dtype=None):
        a = np.asarray(self.phases)
        return a.astype(dtype) if dtype is not None else a

    def __len__(self) -> int:
        return len(self.phases)


def _trim_trailing_zeros(coeffs: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Drop trailing (near-)zero coefficients so the inferred degree is true.

    Always keeps at least one coefficient.
    """
    nz = np.nonzero(np.abs(coeffs) > tol)[0]
    if nz.size == 0:
        return coeffs[:1]
    return coeffs[: nz[-1] + 1]


def _coerce_target(poly: PolyLike, degree: "int | None", basis: str = "chebyshev"):
    """Return (callable_target, degree) from a polynomial spec.

    Accepts:

    * a bare callable -- ``degree`` is required and ``basis`` is ignored;
    * a numpy ``Polynomial`` or ``Chebyshev`` object -- honored as-is in its own
      basis (``basis`` is ignored);
    * a sequence of coefficients -- interpreted in ``basis`` (one of
      ``"chebyshev"`` (DEFAULT, the QSP/QSVT field convention) or ``"monomial"``,
      ascending order).

    Trailing near-zero coefficients are trimmed before inferring the degree.
    """
    # numpy Chebyshev: honor its own (Chebyshev) basis.
    if isinstance(poly, np.polynomial.Chebyshev):
        deg = int(poly.degree()) if degree is None else int(degree)
        return (lambda x: float(poly(x))), deg

    # numpy Polynomial: honor its own (power/monomial) basis.
    if isinstance(poly, np.polynomial.Polynomial):
        deg = int(poly.degree()) if degree is None else int(degree)
        return (lambda x: float(poly(x))), deg

    if callable(poly):
        if degree is None:
            raise ValueError("degree is required when the target is a callable")
        return (lambda x: float(poly(x))), int(degree)

    if basis not in ("chebyshev", "monomial"):
        raise ValueError(
            f"basis must be 'chebyshev' or 'monomial', got {basis!r}"
        )

    coeffs = np.asarray(poly, dtype=float).ravel()
    if coeffs.size == 0:
        raise ValueError("empty coefficient sequence")
    coeffs = _trim_trailing_zeros(coeffs)

    if basis == "chebyshev":
        p = np.polynomial.Chebyshev(coeffs)
    else:
        p = np.polynomial.Polynomial(coeffs)
    deg = int(coeffs.size - 1) if degree is None else int(degree)
    return (lambda x: float(p(x))), deg


def _validate_target(func: Callable[[float], float], degree: int) -> None:
    """Sanity-check a target before solving.

    Raises ``ValueError`` if the target exceeds ``|f| <= 1`` on a grid of
    [-1, 1], or if its parity does not match ``degree % 2`` (even degree must be
    an even function, odd degree an odd function).
    """
    grid = np.linspace(-1.0, 1.0, 257)
    vals = np.array([float(func(float(x))) for x in grid])

    max_abs = float(np.max(np.abs(vals)))
    if max_abs > 1.0 + 1e-9:
        raise ValueError(
            "target violates |f(x)| <= 1 on [-1, 1]: "
            f"max|f| = {max_abs:.6g} (> 1). QSP can only realise polynomials "
            "bounded by 1 in magnitude; rescale the target (e.g. multiply by a "
            "factor < 1) before solving."
        )

    # Parity check: f(-x) should equal +f(x) (even, deg even) or -f(x) (odd).
    want_even = (degree % 2) == 0
    sign = 1.0 if want_even else -1.0
    # Sample symmetric pairs off zero (skip the midpoint, which is trivially
    # self-symmetric).
    xs = np.linspace(0.0, 1.0, 129)[1:]
    pos = np.array([float(func(float(x))) for x in xs])
    neg = np.array([float(func(float(-x))) for x in xs])
    scale = max(1.0, float(np.max(np.abs(pos))))
    parity_err = float(np.max(np.abs(neg - sign * pos))) / scale
    if parity_err > 1e-6:
        want = "even" if want_even else "odd"
        raise ValueError(
            f"target parity mismatch: degree={degree} requires an {want} "
            f"function (f(-x) = {'+' if want_even else '-'}f(x)), but the "
            f"sampled parity error is {parity_err:.3g}. Check that the target's "
            "parity matches degree % 2, or pass the correct degree."
        )


def target2angles(
    func: Callable[[float], float],
    degree: int,
    validate: bool = True,
    tol: float = _DEFAULT_TOL,
    method: str = _DEFAULT_METHOD,
) -> AngleResult:
    """Solve for phases approximating an arbitrary real callable ``func``.

    ``func`` must satisfy ``|func(x)| <= 1`` on [-1, 1] and have parity
    ``degree mod 2`` (even degree -> even function, odd degree -> odd function).

    Args:
        func: real target callable on [-1, 1].
        degree: degree ``d`` of the target polynomial.
        validate: when True (default), check ``|f| <= 1`` and parity on a grid
            and raise ``ValueError`` with an actionable message on violation.
        tol: convergence tolerance; ``AngleResult.converged`` is set to
            ``residual < tol`` (default ``1e-6``). The C++ solver always reports
            the raw worst-case residual.
        method: angle solver to use. ``"sym_qsp"`` (DEFAULT) is the
            symmetric-QSP Newton method -- strictly faster than the homotopy
            solver and reaching far higher degree (1000+) at machine precision.
            ``"homotopy"`` is the homotopy-continuation fallback. Both return
            phases in the same Wx / Re convention.

    Raises:
        ValueError: if ``method`` is not one of ``"sym_qsp"`` / ``"homotopy"``,
            or (when ``validate``) if the target violates ``|f| <= 1`` or parity.
    """
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {sorted(_METHODS)!r}, got {method!r}"
        )
    d = int(degree)
    if validate:
        _validate_target(func, d)
    solve = getattr(_core, _METHODS[method])
    res = solve(lambda x: float(func(x)), d)
    residual = float(res["residual"])
    return AngleResult(
        phases=np.asarray(res["phases"], dtype=float),
        residual=residual,
        converged=bool(residual < tol),
    )


def poly2angles(
    poly: PolyLike,
    degree: "int | None" = None,
    basis: str = "chebyshev",
    validate: bool = True,
    tol: float = _DEFAULT_TOL,
    method: str = _DEFAULT_METHOD,
) -> AngleResult:
    """Solve for phases from a polynomial spec.

    Args:
        poly: a callable, a numpy ``Polynomial``/``Chebyshev`` object, or a
            sequence of coefficients.
        degree: degree override (required for a bare callable; inferred
            otherwise).
        basis: ``"chebyshev"`` (DEFAULT, the QSP/QSVT field convention) or
            ``"monomial"`` for coefficient inputs. A numpy ``Polynomial`` or
            ``Chebyshev`` object is honored in its own basis; a bare callable is
            unaffected.
        validate: forwarded to :func:`target2angles` (default True).
        tol: forwarded to :func:`target2angles` (default ``1e-6``).
        method: forwarded to :func:`target2angles`. ``"sym_qsp"`` (DEFAULT, the
            symmetric-QSP Newton method) or ``"homotopy"`` (the fallback).
    """
    func, deg = _coerce_target(poly, degree, basis=basis)
    return target2angles(func, deg, validate=validate, tol=tol, method=method)


def response(x: float, phases: Sequence[float]) -> float:
    """Achieved QSP polynomial ``Re<0|U(x)|0>`` for the given phases."""
    return float(_core.response(float(x), [float(p) for p in np.asarray(phases).ravel()]))


def QuantumSignalProcessingPhases(
    poly: PolyLike,
    signal_operator: str = "Wx",
    basis: str = "chebyshev",
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper returning just this package's phase array for ``poly``.

    This is named after ``pyqsp.angle_sequence.QuantumSignalProcessingPhases``
    for familiarity, but it is **NOT a drop-in replacement** for pyqsp. It
    returns *this package's* full symmetric phase sequence in the **Wx
    convention** -- a single numpy ndarray of length ``d + 1``. The phase count
    and convention differ from pyqsp (which returns a reduced phase list, and
    whose ``sym_qsp`` path returns a 3-tuple), so the outputs are not
    interchangeable. Only the ``Wx`` signal operator is supported.

    It solves with this package's default ``method="sym_qsp"`` (the
    symmetric-QSP Newton method); pass ``method="homotopy"`` via ``**kwargs`` to
    use the homotopy fallback instead.

    Args:
        poly: a callable, numpy ``Polynomial``/``Chebyshev``, or coefficient
            sequence. Coefficient inputs default to the Chebyshev basis (pass
            ``basis="monomial"`` for the power basis).
        signal_operator: must be ``"Wx"`` (this solver's native convention);
            anything else raises ``NotImplementedError``.
        basis: ``"chebyshev"`` (DEFAULT) or ``"monomial"`` for coefficient
            inputs.

    Returns:
        numpy ndarray of ``d + 1`` full symmetric phases (Wx convention).

    Raises:
        NotImplementedError: if ``signal_operator != "Wx"``.
        RuntimeError: if the solve does not converge (this wrapper discards the
            full result object, so a silent non-converged return would be
            unsafe).
    """
    if signal_operator != "Wx":
        raise NotImplementedError(
            f"signal_operator={signal_operator!r} is not supported; only 'Wx'."
        )
    result = poly2angles(poly, basis=basis, **kwargs)
    if not result.converged:
        raise RuntimeError(
            "QSP angle solve did not converge "
            f"(residual = {result.residual:.3g}). The target may be too close "
            "to |f| = 1, too high-degree, or mis-specified. Use poly2angles() "
            "to inspect the full AngleResult (phases + residual) instead."
        )
    return np.asarray(result.phases, dtype=float)
