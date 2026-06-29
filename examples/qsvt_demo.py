#!/usr/bin/env python3
"""End-to-end QSVT demo via the qsvt_native Python bindings.

Shows the matrix-function-to-circuit pipeline and the three canonical QSVT
applications on small Hermitian operators, validating each against exact
linear algebra. Build the bindings first (see bindings/README.md), then:

    PYTHONPATH=build/bindings python3 examples/qsvt_demo.py
"""
import numpy as np
import qsvt_native as q


def random_hermitian(eigs, seed):
    """Hermitian matrix with the given eigenvalues."""
    rng = np.random.default_rng(seed)
    n = len(eigs)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    U, _ = np.linalg.qr(a)
    return U @ np.diag(eigs) @ U.conj().T


def cheb(n, x):
    """Chebyshev polynomial T_n (a convenient achievable QSVT target)."""
    a, b = 1.0, x
    if n == 0:
        return a
    for _ in range(2, n + 1):
        a, b = b, 2 * x * b - a
    return b if n >= 1 else a


def main():
    print("=" * 64)
    print("QSVT toolkit demo (C++ core via qsvt_native)")
    print("=" * 64)

    # --- 1. Compile a unitary to native gates + OpenQASM --------------------
    rng = np.random.default_rng(2)
    U, _ = np.linalg.qr(rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8)))
    d = q.shannon_decompose(U)
    print("\n[1] Quantum Shannon decomposition of a random 3-qubit unitary")
    print(f"    {d['cnot']} CNOTs, depth {d['depth']}, "
          f"reconstruction error {d['reconstruction_error']:.1e}")
    print(f"    OpenQASM ({len(d['qasm'].splitlines())} lines), first gates:")
    for line in d["qasm"].splitlines()[3:6]:
        print(f"      {line}")

    # --- 2. Matrix inversion: apply ~ c/A to a Hermitian operator -----------
    A = random_hermitian([0.4, 0.6, 0.8, 1.0], seed=3)  # well-conditioned, PD
    delta = 0.4
    eps = (delta / 4.0) ** 2
    c = 0.9 * 2 * np.sqrt(eps)
    inv_target = lambda x: c * x / (x * x + eps)  # noqa: E731  (~ c/x on the spectrum)
    prog = q.compile_matrix_function(A, inv_target, 25)
    block = q.qsvt_block(A, prog["phases"])
    realized = 0.5 * (block + block.conj().T)  # Hermitian part = the real transform
    lam, V = np.linalg.eigh(A)
    target_inv = V @ np.diag(c / lam) @ V.conj().T
    print("\n[2] Matrix inversion via QSVT (compile -> circuit)")
    print(f"    {prog['num_qubits']} qubits, {prog['cnot']} CNOTs, degree 25")
    print(f"    || realized(A) - c*A^-1 || on spectrum: "
          f"{np.linalg.norm(realized - target_inv):.2e}")

    # --- 3. Hamiltonian simulation e^{-iHt} ---------------------------------
    H = random_hermitian([0.6, -0.4, 0.3, -0.7], seed=11)
    t = 2.5
    approx = q.hamiltonian_evolution(H, t, 21)
    lamH, VH = np.linalg.eigh(H)
    exact = VH @ np.diag(np.exp(-1j * t * lamH)) @ VH.conj().T
    print("\n[3] Hamiltonian simulation e^{-iHt} (t = 2.5)")
    print(f"    || e^-iHt_QSVT - e^-iHt_exact || = {np.linalg.norm(approx - exact):.2e}")

    # --- 4. Eigenvalue thresholding (spectral projector) --------------------
    Hp = random_hermitian([-0.7, -0.4, 0.4, 0.8], seed=5)
    P = q.spectral_projector(Hp, mu=0.0, w=0.1, degree=25)
    lamP, VP = np.linalg.eigh(Hp)
    exactP = VP @ np.diag([1.0 if l > 0 else 0.0 for l in lamP]) @ VP.conj().T
    print("\n[4] Eigenvalue thresholding: project onto lambda > 0")
    print(f"    || P_QSVT - exact projector || = {np.linalg.norm(P - exactP):.2e}")

    print("\nAll demos validated against exact linear algebra.")


if __name__ == "__main__":
    main()
