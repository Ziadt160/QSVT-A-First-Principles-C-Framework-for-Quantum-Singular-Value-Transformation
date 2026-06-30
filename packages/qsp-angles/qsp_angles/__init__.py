"""qsp-angles: a fast C++ QSP/QSVT phase-factor (angle) solver.

The heavy lifting is a self-contained C++ solver (homotopy continuation + an
exact analytic Jacobian, Levenberg-Marquardt over Chebyshev nodes) exposed to
Python via pybind11. It computes the phase sequence ``Phi`` such that, in the
**Wx convention**,

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

pyqsp-compatible entry point
----------------------------
>>> import numpy as np
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


@dataclass
class AngleResult:
    """Result of an angle solve.

    Attributes:
        phases: length ``degree + 1`` symmetric phase sequence (Wx convention).
        residual: worst-case ``|Re<0|U|0> - f|`` over a fine grid on [-1, 1].
        converged: whether ``residual < 1e-6``.
    """

    phases: np.ndarray
    residual: float
    converged: bool

    def __array__(self, dtype=None):
        a = np.asarray(self.phases)
        return a.astype(dtype) if dtype is not None else a

    def __len__(self) -> int:
        return len(self.phases)


def _coerce_target(poly: PolyLike, degree: "int | None"):
    """Return (callable_target, degree) from a polynomial spec.

    Accepts a callable (degree required), a numpy Polynomial, or a sequence of
    coefficients in the **monomial** basis (ascending order), matching numpy's
    ``Polynomial`` convention.
    """
    # numpy Polynomial (or anything exposing a numeric degree() + __call__).
    if isinstance(poly, np.polynomial.Polynomial):
        deg = int(poly.degree()) if degree is None else int(degree)
        return (lambda x: float(poly(x))), deg

    if callable(poly):
        if degree is None:
            raise ValueError("degree is required when the target is a callable")
        return (lambda x: float(poly(x))), int(degree)

    coeffs = np.asarray(poly, dtype=float).ravel()
    if coeffs.size == 0:
        raise ValueError("empty coefficient sequence")
    p = np.polynomial.Polynomial(coeffs)
    deg = int(coeffs.size - 1) if degree is None else int(degree)
    return (lambda x: float(p(x))), deg


def target2angles(func: Callable[[float], float], degree: int) -> AngleResult:
    """Solve for phases approximating an arbitrary real callable ``func``.

    ``func`` must satisfy ``|func(x)| <= 1`` on [-1, 1] and have parity
    ``degree mod 2`` (even degree -> even function, odd degree -> odd function).
    """
    d = _core.poly_to_angles(lambda x: float(func(x)), int(degree))
    return AngleResult(
        phases=np.asarray(d["phases"], dtype=float),
        residual=float(d["residual"]),
        converged=bool(d["converged"]),
    )


def poly2angles(poly: PolyLike, degree: "int | None" = None) -> AngleResult:
    """Solve for phases from a polynomial spec (coeffs, numpy Polynomial, or callable)."""
    func, deg = _coerce_target(poly, degree)
    return target2angles(func, deg)


def response(x: float, phases: Sequence[float]) -> float:
    """Achieved QSP polynomial ``Re<0|U(x)|0>`` for the given phases."""
    return float(_core.response(float(x), [float(p) for p in np.asarray(phases).ravel()]))


def QuantumSignalProcessingPhases(
    poly: PolyLike,
    signal_operator: str = "Wx",
    **kwargs,
) -> np.ndarray:
    """pyqsp-compatible alias: return just the phase array for ``poly``.

    Mirrors ``pyqsp.angle_sequence.QuantumSignalProcessingPhases`` for the common
    call shape so existing pyqsp code can swap the import. Only the ``Wx``
    convention is supported (the native convention of this solver). Extra keyword
    arguments accepted by pyqsp are ignored for compatibility.

    Note: this returns the symmetric phase sequence in the Wx convention; it is
    API-compatible, not guaranteed bit-identical to pyqsp's output.
    """
    if signal_operator != "Wx":
        raise NotImplementedError(
            f"signal_operator={signal_operator!r} is not supported; only 'Wx'."
        )
    return np.asarray(poly2angles(poly).phases, dtype=float)
