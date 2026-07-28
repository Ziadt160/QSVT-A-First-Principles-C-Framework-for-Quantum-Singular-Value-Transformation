"""Advection-diffusion: a NON-SYMMETRIC operator, where the transform kernels
cannot run at all and QSVT still can.

The previous step (variable_coefficient.py) removed the analytic eigenbasis while
keeping the operator symmetric. This removes symmetry itself:

    du/dt = alpha d2u/dx2 - v du/dx

With central differencing the advection term is ANTISYMMETRIC, so the implicit-Euler
matrix A = I + dt(alpha L_diff + v L_adv) is non-symmetric and, at high Peclet
number, strongly non-normal (A A^dag != A^dag A).

Why the transform kernels are finished here, and not merely slower:
  * the discrete sine transform does not diagonalise a non-symmetric operator;
  * a non-symmetric tridiagonal Toeplitz matrix IS diagonalisable in closed form,
    but its eigenvector matrix is a diagonal scaling times the sine matrix -- it is
    NOT orthogonal, so the similarity transform is not unitary and there is no
    circuit that implements it. QSM/Schade rely on a unitary (DST) transform.

QSVT is untouched because it transforms SINGULAR values, not eigenvalues, and
A^-1 = V Sigma^-1 U^dag holds for any invertible A. An odd polynomial p ~ c/x
applied to the singular values yields V p(Sigma) U^dag ~ c A^-1 directly.

This file simulates the singular-value transform densely (via SVD), exactly as
qsvt_block simulates the Hermitian case densely via eigh -- and as the benchmark's
own statevector path does. It is not a circuit execution.

    python advection_diffusion.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import qsp_bridge as Q  # noqa: E402
from numpy.polynomial import chebyshev as Cheb  # noqa: E402

N = 64
EPS = 1e-3
SAFETY = 0.95


def build_operator(n, peclet, r=0.4):
    """A = I + r*(L_diff + Pe * L_adv), Dirichlet BC, central differences.

    Peclet = v*dx/alpha is the cell Peclet number; Pe = 0 recovers the symmetric
    heat operator the benchmark uses.
    """
    Ld = (np.diag(2.0 * np.ones(n))
          + np.diag(-1.0 * np.ones(n - 1), 1)
          + np.diag(-1.0 * np.ones(n - 1), -1))
    La = (np.diag(0.5 * np.ones(n - 1), 1)      # central difference: antisymmetric
          - np.diag(0.5 * np.ones(n - 1), -1))
    return np.eye(n) + r * (Ld + peclet * La)


def sv_transform(A_norm, coeffs):
    """Dense simulation of the QSVT singular-value transform.

    For an odd polynomial p, QSVT applied to a block encoding of A yields
    V p(Sigma) U^dag. With p(x) ~ c/x this is ~ c A^-1, for ANY invertible A --
    no symmetry required.
    """
    U, S, Vh = np.linalg.svd(A_norm)
    pS = Cheb.chebval(S, coeffs)
    return (Vh.conj().T * pS) @ U.conj().T


def main():
    rng = np.random.default_rng(11)
    b = rng.standard_normal(N)
    b /= np.linalg.norm(b)

    # Sanity: at Pe = 0 the operator is symmetric, so the SV route must agree
    # with the Hermitian pipeline this project already validated.
    A0 = build_operator(N, 0.0)
    A0n = A0 / np.linalg.svd(A0, compute_uv=False)[0]
    k0 = float(np.linalg.cond(A0n))
    d0, _ = Q.min_degree(k0, EPS)
    c0, _c, _r = Q.approx_inverse(k0, d0)
    x_sv = sv_transform(A0n, c0 * SAFETY) @ b
    ph, _res = Q.solve_phases((c0 * SAFETY).tolist())
    blk = Q.qsvt_block(A0n, ph)
    x_h = np.real((blk + blk.conj().T) / 2) @ b
    agree = np.linalg.norm(x_sv / np.linalg.norm(x_sv) - x_h / np.linalg.norm(x_h))
    print(f"sanity, Pe=0: SV route vs Hermitian pipeline agree to {agree:.2e}\n")

    print(f"Advection-diffusion, {N} points, implicit Euler (r=0.4).")
    print("Pe = cell Peclet number; Pe=0 is the benchmark's symmetric heat operator.\n")
    print(f"{'Pe':>6} {'kappa':>8} {'non-normal':>11} {'DST diag?':>10} | "
          f"{'QSVT d=15':>10} {'QSVT auto':>10} {'deg':>5}")
    print("-" * 76)

    j = np.arange(1, N + 1)
    S = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(j, j) * np.pi / (N + 1))

    for pe in (0.0, 1.0, 5.0, 20.0, 100.0):
        A = build_operator(N, pe)
        smax = np.linalg.svd(A, compute_uv=False)[0]
        An = A / smax
        sv = np.linalg.svd(An, compute_uv=False)
        kappa = float(sv[0] / sv[-1])

        # Non-normality: ||A A^H - A^H A|| / ||A||^2. Zero iff normal.
        comm = An @ An.conj().T - An.conj().T @ An
        nonnormal = float(np.linalg.norm(comm) / np.linalg.norm(An) ** 2)

        M = S.T @ An @ S
        offdiag = float(np.linalg.norm(M - np.diag(np.diag(M))) / np.linalg.norm(M))

        x_true = np.linalg.solve(An, b)
        x_true /= np.linalg.norm(x_true)

        def run(deg):
            co, _c, _r = Q.approx_inverse(kappa, deg)
            x = sv_transform(An, co * SAFETY) @ b
            n = np.linalg.norm(x)
            x = x / n if n > 0 else x
            return float(np.linalg.norm(x - x_true))

        err15 = run(15)
        deg, _ = Q.min_degree(kappa, EPS)
        erra = run(deg)

        print(f"{pe:>6.0f} {kappa:>8.1f} {nonnormal:>11.2e} {offdiag:>10.2e} | "
              f"{err15:>10.2e} {erra:>10.2e} {deg:>5}")

    print("\nnon-normal = ||AA^H - A^H A|| / ||A||^2, zero only for normal operators.")
    print("Once it is non-zero the operator has no unitary diagonalisation, so a")
    print("DST-based kernel has nothing to implement -- it is inapplicable, not slow.")
    print("QSVT transforms singular values, so A^-1 = V Sigma^-1 U^dag still works.")


if __name__ == "__main__":
    main()
