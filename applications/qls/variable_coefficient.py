"""Variable-coefficient diffusion: the regime where the spectral shortcut dies.

a published quantum-PDE benchmark solves the CONSTANT-coefficient heat equation, whose
Dirichlet Laplacian is diagonalised exactly by the discrete sine transform. Two of
their kernels (QSM, Schade-Hamiltonian) exploit that and come out bitwise identical
to the reference -- they are not approximating the answer, they are recomputing it
by the same route. No polynomial method can beat a closed form.

Heterogeneous media break it. For

    du/dt = d/dx( alpha(x) du/dx ),   alpha piecewise constant,

the operator is still SYMMETRIC -- so a Hermitian QSVT pipeline is unchanged -- but
it has NO analytic eigenbasis. A DST-based spectral kernel would first need a
classical eigendecomposition, which is circular: it needs the answer to compute the
answer. QSVT needs only a block encoding, so it is unaffected.

The contrast ratio C = alpha_max/alpha_min also sets the conditioning, so the same
knob that kills the spectral shortcut drives the problem into the ill-conditioned
regime the study explicitly excludes (their SS II).

Three solvers on the identical operator:
  * spectral (DST)   -- what QSM/Schade do, applied to an operator that is not
                        sine-diagonal
  * QSVT, degree 15  -- the fixed degree the benchmark's QSVT kernel hard-codes
  * QSVT, auto degree -- degree chosen from the measured conditioning

Framing note, stated rather than hidden: choosing a problem one's own method wins
is only legitimate if the problem is standard and the reason is structural. Layered
diffusion is textbook, and the reason the spectral kernels fail is that the
eigenbasis is not known in closed form -- not that they are slow. QSVT wins here on
APPLICABILITY, not efficiency; it still pays ~kappa^3 queries per accepted sample.

    python variable_coefficient.py
"""

import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as Cheb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import qsp_bridge as Q  # noqa: E402

N = 64                 # interior points
EPS = 1e-3             # QSVT target accuracy
SAFETY = 0.95          # sup-norm shave (see gap_cost_curve.py for why not 0.999)


def alpha_profile(n, contrast, layers=4):
    """Piecewise-constant diffusivity: `layers` slabs alternating 1 and contrast."""
    a = np.ones(n + 1)
    edge = np.linspace(0, n + 1, layers + 1).astype(int)
    for k in range(layers):
        if k % 2:
            a[edge[k]:edge[k + 1]] = contrast
    return a


def build_operator(n, contrast, r=0.4):
    """Implicit-Euler matrix A = I + r*L for d/dx(alpha du/dx), Dirichlet BC.

    Harmonic-mean face coefficients keep A symmetric positive definite, which is
    the standard finite-volume discretisation for heterogeneous media.
    """
    a = alpha_profile(n, contrast)
    face = np.zeros(n + 1)
    for i in range(n + 1):
        al, ar = a[i], a[min(i + 1, n)]
        face[i] = 2.0 * al * ar / (al + ar)          # harmonic mean
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = face[i] + face[i + 1]
        if i > 0:
            L[i, i - 1] = -face[i]
        if i < n - 1:
            L[i, i + 1] = -face[i + 1]
    L /= face.mean()                                  # keep r comparable to theirs
    return np.eye(n) + r * L


def dst_matrix(n):
    """Orthonormal discrete sine transform -- the basis QSM/Schade assume."""
    j = np.arange(1, n + 1)
    S = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(j, j) * np.pi / (n + 1))
    return S


def qsvt_solve(A_norm, b, degree, kappa):
    """x ~ A^-1 b via QSVT with an explicit polynomial degree."""
    coeffs, c, _rel = Q.approx_inverse(kappa, degree)
    coeffs = coeffs * SAFETY
    phases, resid = Q.solve_phases(coeffs.tolist())
    block = Q.qsvt_block(A_norm, phases)
    fA = np.real((block + block.conj().T) / 2)        # = c * A_norm^-1
    x = fA @ b
    nx = np.linalg.norm(x)
    return (x / nx if nx > 0 else x), resid


def main():
    S = dst_matrix(N)
    rng = np.random.default_rng(7)
    b = rng.standard_normal(N)
    b /= np.linalg.norm(b)

    print(f"Layered diffusion, {N} interior points, implicit Euler (r=0.4).")
    print("Contrast C = alpha_max/alpha_min. C=1 recovers the constant-coefficient case.\n")
    print(f"{'C':>6} {'kappa':>8} {'DST diag?':>10} | {'spectral':>10} "
          f"{'QSVT d=15':>10} {'QSVT auto':>10} {'deg':>5}")
    print("-" * 72)

    for contrast in (1, 10, 100, 1000):
        A = build_operator(N, contrast)
        nrm = float(np.linalg.eigvalsh(A)[-1])
        An = A / nrm
        w = np.linalg.eigvalsh(An)
        kappa = float(w[-1] / w[0])

        # How nearly does the sine basis diagonalise A? Off-diagonal mass of S^T A S.
        M = S.T @ An @ S
        offdiag = float(np.linalg.norm(M - np.diag(np.diag(M))) / np.linalg.norm(M))

        x_true = np.linalg.solve(An, b)
        x_true /= np.linalg.norm(x_true)

        # (1) spectral: assume A = S diag(d) S^T, invert in that basis.
        d = np.diag(M)
        x_dst = S @ ((S.T @ b) / d)
        x_dst /= np.linalg.norm(x_dst)
        err_dst = float(np.linalg.norm(x_dst - x_true))

        # (2) QSVT at the benchmark's hard-coded degree 15.
        x15, _r15 = qsvt_solve(An, b, 15, kappa)
        err15 = float(np.linalg.norm(x15 - x_true))

        # (3) QSVT with the degree the conditioning actually requires.
        deg, _ = Q.min_degree(kappa, EPS)
        xa, resid = qsvt_solve(An, b, deg, kappa)
        erra = float(np.linalg.norm(xa - x_true))

        print(f"{contrast:>6} {kappa:>8.1f} {offdiag:>10.2e} | {err_dst:>10.2e} "
              f"{err15:>10.2e} {erra:>10.2e} {deg:>5}"
              + ("" if resid < 1e-9 else f"  (angle resid {resid:.1e})"))

    print("\n'DST diag?' is the off-diagonal mass of S^T A S: 0 means the sine basis")
    print("diagonalises A exactly (their case), and the spectral kernels are exact.")
    print("Once it is non-zero there is no analytic eigenbasis to exploit, and a")
    print("spectral kernel would need a classical eigendecomposition to proceed.")


if __name__ == "__main__":
    main()
