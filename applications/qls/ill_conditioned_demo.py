"""The result the that study harness is missing: Paradigm-A QSVT in the
ILL-CONDITIONED regime their benchmark excludes.

Their heat benchmark fixes the CFL ratio r = alpha*dt/dx^2 = 0.4, giving
kappa = 1+4r ~ 2.6 (well-conditioned) and explicitly *not* engaging the
amplitude-amplification / preconditioning refinements for ill-conditioned A
(paper §II). Implicit-Euler schemes are unconditionally stable, so taking a
LARGE implicit step (r >> 1) is legitimate physics -- and it drives
kappa = 1+4r into the tens/hundreds. That is the regime where:
  * the 1/x polynomial degree explodes (hundreds-thousands),
  * pyqsp's angle finding becomes slow while our sym_qsp stays fast, and
  * a direct SPARSE block-encoding stays compilable where their Fourier-LCU
    QLS route is uncompilable at n>=5.

This script runs THEIR heat matrix family A = I + r*L0 at large r through THIS
project's dense QSVT linear-solve pipeline (angles from our zero-dep sym_qsp C
ABI, timed), and reports that the solve stays high-fidelity. No simulator.

    python ill_conditioned_demo.py
"""

import time

import numpy as np

from qls_dense import ensure_solver, solve_phases, qsvt_block
from heat_operator import heat_matrix
from qsp_bridge import min_degree, approx_inverse

SAFETY = 0.999


def pauli_term_count(M, tol=1e-9):
    """Number of nonzero Pauli-string coefficients of a 2^n x 2^n matrix.

    This is L, the term count of a direct Pauli-LCU block-encoding of M. Cheap
    for n<=6 (4^n strings). Compared against the 4^n cost of dense synthesis.
    """
    n = int(round(np.log2(M.shape[0])))
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]], complex)
    paulis = [I2, X, Y, Z]
    count = 0
    for idx in range(4 ** n):
        digits = []
        t = idx
        for _ in range(n):
            digits.append(t & 3); t >>= 2
        P = np.array([[1.0]], complex)
        for d in digits:
            P = np.kron(P, paulis[d])
        coeff = np.trace(P.conj().T @ M) / (2 ** n)
        if abs(coeff) > tol:
            count += 1
    return count


def main():
    lib = ensure_solver()
    rng_b = np.random.default_rng(11)

    # r chosen so kappa_inf = 1+4r hits ~10 / 30 / 60 / 80 / 100 (stiff implicit
    # steps). The sym_qsp Newton core converges to machine-residual angles for
    # these 1/x targets through degree ~900+ in a few seconds each. (An earlier
    # apparent "loss of convergence" past degree ~600 was NOT a solver limit: it
    # was an approximation-side bug -- approx_inverse fit c/x on a uniform grid
    # and measured its sup on a uniform grid, both ill-conditioned at high degree,
    # so it returned polynomials with true sup-norm 10-300, i.e. invalid QSP
    # targets |p|>>1 that no phase sequence can realise. Fixed by fitting and
    # measuring on Chebyshev-clustered nodes; the solver was always fine.)
    regimes = [(2.25, 10), (7.25, 30), (14.75, 60), (19.75, 80), (24.75, 100)]
    n_list = (4, 5, 6)
    eps = 1e-3

    print("Angle generation (our sym_qsp core, via zero-dep C ABI):")
    print(f"{'kappa~':>7} {'degree':>7} {'poly_relerr':>12} {'sym_qsp_time_s':>15} {'angle_resid':>12}")
    angle_cache = {}
    for r, ktar in regimes:
        kappa = 1.0 + 4.0 * r  # sup over n; a valid QSP target for every finite-N kappa<=this
        degree, _rel = min_degree(kappa, eps)
        coeffs, c, relp = approx_inverse(kappa, degree)
        coeffs = coeffs * SAFETY
        c *= SAFETY
        t0 = time.perf_counter()
        phases, ares = solve_phases(lib, coeffs.tolist())
        dt = time.perf_counter() - t0
        angle_cache[ktar] = (r, kappa, degree, coeffs, c, phases)
        print(f"{kappa:>7.1f} {degree:>7} {relp:>12.2e} {dt:>15.3f} {ares:>12.1e}")

    print("\nDense QSVT linear solve on the stiff heat matrix (their family, large r):")
    print(f"{'n':>3} {'N':>5} {'kappa(A)':>9} {'degree':>7} {'op_relerr':>11} "
          f"{'fidelity':>10} {'succ_prob':>10}")
    for r, ktar in regimes:
        _r, _kinf, degree, coeffs, c, phases = angle_cache[ktar]
        for n in n_list:
            A = heat_matrix(n, r=r)
            normA = np.linalg.eigvalsh(A)[-1]
            At = A / normA                       # eigenvalues in [1/kappa, 1]
            kappaA = np.linalg.cond(At)
            block = qsvt_block(At, phases)
            fA = (block + block.conj().T) / 2     # c * At^{-1} (LCU, +1 ancilla)

            Ainv = np.linalg.inv(At)
            op_relerr = np.linalg.norm(fA - c * Ainv, 2) / np.linalg.norm(c * Ainv, 2)

            b = rng_b.standard_normal(2 ** n)
            b /= np.linalg.norm(b)
            x_exact = Ainv @ b; x_exact /= np.linalg.norm(x_exact)
            xq = fA @ b
            succ = np.linalg.norm(xq) ** 2
            xq /= np.linalg.norm(xq)
            fid = abs(np.vdot(x_exact, xq))
            print(f"{n:>3} {2**n:>5} {kappaA:>9.2f} {degree:>7} {op_relerr:>11.2e} "
                  f"{fid:>10.6f} {succ:>10.2e}")

    print("\nBlock-encoding cost of the heat matrix A = I + r*L0:")
    print(f"{'n':>3} {'N':>5} {'direct Pauli-LCU L':>18} {'dense synth ~0.58*4^n':>22}")
    for n in n_list:
        A = heat_matrix(n, r=regimes[0][0])
        L = pauli_term_count(A)
        dense = 0.58 * (4 ** n)
        print(f"{n:>3} {2**n:>5} {L:>18} {dense:>22.0f}")
    print("  direct Pauli-LCU is L = 2^n = O(N) simple strings (measured): far below")
    print("  dense 4^n and vastly below their Fourier-LCU (~1e6 at n=4, uncompilable")
    print("  at n>=5, §V-A). A is 2-sparse (tridiagonal) => a sparse-access oracle")
    print("  block-encoding is the poly(n) route; the naive Pauli-LCU here is not poly(n).")


if __name__ == "__main__":
    main()
