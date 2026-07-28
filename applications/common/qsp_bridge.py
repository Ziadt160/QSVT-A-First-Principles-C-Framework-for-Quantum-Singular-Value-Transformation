"""Self-contained sym_qsp angle-solver helpers for QSVTSymQspKernel.

Frozen from the qsp-angles project so the kernel has no external path dependency:
  * approx_inverse / min_degree -- near-minimax odd 1/x Chebyshev fit on
    [1/kappa, 1], with fit nodes AND sup-norm measured on CHEBYSHEV-clustered
    nodes. (A uniform grid is ill-conditioned above degree ~600: the fit
    oscillates to O(10^2) between nodes near x=+-1 and a uniform sup grid steps
    over those spikes, silently producing |p|>>1 -- an invalid QSP target that no
    phase sequence can realise. Chebyshev nodes keep the target valid and the
    UNMODIFIED sym_qsp Newton solver reaches machine residual past degree 1800.)
  * solve_phases -- ctypes driver for the zero-dependency C ABI of the
    Dong-Lin-Ni-Wang symmetric-QSP Newton solver (no pyqsp, no python3-dev).
  * qsvt_block -- top-left block of the QSVT unitary U_Phi (matches the project's
    QsvtPipeline.cpp: U_W=[[A,iB],[iB,A]], E(phi)=diag(e^{i phi}I, e^{-i phi}I)).
  * solve_angles -- robust wrapper: minimal degree for target eps, verified, and
    degree-bumped if the (low-degree) solve is ever poor.

The shared library is built on first use from ../../packages/qsp-angles (this
repo). When dropping this file into another project (e.g. HelloQuantum), set
QSP_ANGLES_SO to a prebuilt libqsp_angles_c.so, or QSP_ANGLES_PKG to a checkout
of packages/qsp-angles.
"""

import ctypes
import os
import subprocess

import numpy as np
from numpy.polynomial import chebyshev as Cheb

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.environ.get(
    "QSP_ANGLES_PKG",
    os.path.normpath(os.path.join(_HERE, "..", "..", "packages", "qsp-angles")),
)
_SO = os.environ.get("QSP_ANGLES_SO",
                     os.path.join(_HERE, ".build", "libqsp_angles_c.so"))
_LIB = None


def _ensure_so():
    if os.path.exists(_SO):
        return _SO
    os.makedirs(os.path.dirname(_SO), exist_ok=True)
    subprocess.run(
        ["g++", "-O3", "-fPIC", "-shared", "-std=c++17",
         "-I" + os.path.join(_PKG, "capi"), "-I" + os.path.join(_PKG, "src"),
         os.path.join(_PKG, "capi", "qsp_angles.cpp"),
         os.path.join(_PKG, "src", "SymQspAngleSolver.cpp"),
         "-o", _SO],
        check=True,
    )
    return _SO


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(_ensure_so())
        PD = ctypes.POINTER(ctypes.c_double)
        PI = ctypes.POINTER(ctypes.c_int)
        lib.qsp_solve_chebyshev.restype = ctypes.c_int
        lib.qsp_solve_chebyshev.argtypes = [
            ctypes.c_int, PD, ctypes.c_int, PD, ctypes.c_int, PI, PD, PI,
        ]
        _LIB = lib
    return _LIB


def approx_inverse(kappa, degree, npts=6000):
    """Best odd degree-`degree` Chebyshev fit of c/x on [1/kappa, 1], |p|<=1."""
    delta = 1.0 / kappa
    c0 = 1.0 / (2.0 * kappa)
    m = max(npts, 4 * degree + 2)
    u = np.cos(np.linspace(0.0, np.pi, m))
    x = delta + (1.0 - delta) * (u + 1.0) / 2.0
    y = c0 / x
    odd = list(range(1, degree + 1, 2))
    V = Cheb.chebvander(x, degree)
    a, *_ = np.linalg.lstsq(V[:, odd], y, rcond=None)
    coeffs = np.zeros(degree + 1)
    coeffs[odd] = a
    xf = np.cos(np.linspace(0.0, np.pi, 8 * degree + 2))
    sup = float(np.max(np.abs(Cheb.chebval(xf, coeffs))))
    coeffs /= sup
    c = c0 / sup
    xd = np.cos(np.linspace(0.0, np.pi, 4 * degree + 2))
    xd = xd[xd >= delta]
    rel = float(np.max(np.abs(Cheb.chebval(xd, coeffs) - c / xd) / np.abs(c / xd)))
    return coeffs, c, rel


def min_degree(kappa, eps, dmax=6000):
    """Smallest odd degree with rel err <= eps (coarse-to-fine, O(log dmax))."""
    def rel_at(d):
        return approx_inverse(kappa, d)[2]

    lo, hi = 1, 3
    if rel_at(lo) <= eps:
        return lo, rel_at(lo)
    while hi <= dmax and rel_at(hi) > eps:
        lo, hi = hi, hi * 2
    if hi > dmax and rel_at(dmax) > eps:
        return None, None
    hi = min(hi, dmax)
    while hi - lo > 2:
        mid = ((lo + hi) // 2) | 1
        if rel_at(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi, rel_at(hi)


def solve_phases(cheb_coeffs):
    """Return (phases, residual) from the sym_qsp C ABI for parity-reduced coeffs."""
    lib = _lib()
    n = len(cheb_coeffs)
    degree = n - 1
    c = (ctypes.c_double * n)(*cheb_coeffs)
    ph = (ctypes.c_double * n)()
    ln, res, cv = ctypes.c_int(0), ctypes.c_double(0), ctypes.c_int(0)
    st = lib.qsp_solve_chebyshev(
        degree, c, n, ph, n,
        ctypes.byref(ln), ctypes.byref(res), ctypes.byref(cv),
    )
    if st != 0:
        raise RuntimeError(f"qsp_solve_chebyshev status {st}")
    return np.array(ph[: ln.value]), res.value


def solve_angles(kappa, eps, safety=0.999, resid_tol=1e-6):
    """Robust angle solve: minimal degree for `eps`, verified, degree-bumped if poor.

    Returns (phases, c, degree, poly_relerr, angle_resid).
    """
    degree, _ = min_degree(kappa, eps)
    last = None
    for bump in (0, 6, 12, 24, 48):
        d = degree + bump
        coeffs, c, relp = approx_inverse(kappa, d)
        coeffs = coeffs * safety
        phases, resid = solve_phases(coeffs.tolist())
        last = (phases, c * safety, d, relp, resid)
        if resid < resid_tol:
            return last
    return last


def qsvt_block(A, phases):
    """Top-left block of U_Phi = complex P(A). Matches QsvtPipeline.cpp."""
    k = A.shape[0]
    w, V = np.linalg.eigh(A)
    B = (V * np.sqrt(np.clip(1 - w ** 2, 0, None))) @ V.conj().T
    Uw = np.block([[A, 1j * B], [1j * B, A]])
    diag0 = np.concatenate([np.ones(k), np.zeros(k)])

    def E(phi):
        d = np.exp(1j * phi) * diag0 + np.exp(-1j * phi) * (1 - diag0)
        return np.diag(d)

    U = E(phases[0])
    for p in phases[1:]:
        U = U @ Uw @ E(p)
    return U[:k, :k]
