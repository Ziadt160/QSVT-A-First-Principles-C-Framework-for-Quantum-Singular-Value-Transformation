"""W2: dense (exact-linear-algebra) validation of the QSVT quantum-linear-systems
pipeline. NO simulator, NO Qrack -- this proves the *math* of the pipeline end to
end before circuit-level work:

    W1 polynomial p ≈ c/x  ->  sym_qsp angles  ->  QSVT block P(A)  ->  solve A x = b

The QSVT block is built to match the project's verified construction exactly
(src/QsvtPipeline.cpp): U_W = [[A, iB],[iB, A]] with B=sqrt(I-A^2);
U_Phi = E(phi_0) prod[U_W E(phi_k)], E(phi)=diag(e^{i phi}I, e^{-i phi}I); the
block is the top-left corner = the complex P(A). The Hermitian part
f(A) = (P(A)+P(A)†)/2 = (real QSP polynomial)(A) ≈ c A^{-1} is the linear-system
operator (the circuit realises it via an LCU of U_Phi and U_Phi†, +1 ancilla).

Angles come from THIS project's solver through its zero-dependency C ABI (built
here with g++ -- no Eigen needed since the sym_qsp core is dependency-free, no
python3-dev needed since we use ctypes).

    python qls_dense.py
"""

import ctypes
import os
import subprocess

import numpy as np

import oneoverx_approx as approx  # W1

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "..", "packages", "qsp-angles"))
SAFETY = 0.999  # shave the W1 sup-norm strictly below 1 (valid QSP target)


def ensure_solver(lib_path="/tmp/libqsp_angles_c.so"):
    """Build the zero-dep angle solver as a shared lib (g++, no Eigen) and load it."""
    if not os.path.exists(lib_path):
        subprocess.run(
            ["g++", "-O3", "-fPIC", "-shared", "-std=c++17",
             "-I" + os.path.join(PKG, "capi"), "-I" + os.path.join(PKG, "src"),
             os.path.join(PKG, "capi", "qsp_angles.cpp"),
             os.path.join(PKG, "src", "SymQspAngleSolver.cpp"),
             "-o", lib_path],
            check=True,
        )
    lib = ctypes.CDLL(lib_path)
    D, PD, PI = ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.qsp_solve_chebyshev.restype = ctypes.c_int
    lib.qsp_solve_chebyshev.argtypes = [ctypes.c_int, PD, ctypes.c_int, PD, ctypes.c_int, PI, PD, PI]
    return lib


def solve_phases(lib, cheb_coeffs):
    n = len(cheb_coeffs)
    degree = n - 1
    c = (ctypes.c_double * n)(*cheb_coeffs)
    ph = (ctypes.c_double * n)()
    ln, res, cv = ctypes.c_int(0), ctypes.c_double(0), ctypes.c_int(0)
    st = lib.qsp_solve_chebyshev(degree, c, n, ph, n,
                                 ctypes.byref(ln), ctypes.byref(res), ctypes.byref(cv))
    if st != 0:
        raise RuntimeError(f"qsp_solve_chebyshev status {st}")
    return np.array(ph[: ln.value]), res.value


def random_hermitian(n_qubits, kappa, seed):
    """Hermitian A, eigenvalues evenly in [1/kappa, 1] (condition number = kappa)."""
    dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    eig = np.linspace(1.0 / kappa, 1.0, dim)
    G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(G)
    A = (Q * eig) @ Q.conj().T
    return (A + A.conj().T) / 2


def qsvt_block(A, phases):
    """Top-left block of U_Phi = complex P(A). Matches src/QsvtPipeline.cpp."""
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


def main():
    lib = ensure_solver()
    rng_b = np.random.default_rng(7)
    print("n_qubits,kappa,degree,eps,op_relerr,fidelity,success_prob,angle_resid")
    for n_qubits in (1, 2, 3, 4):
        for kappa in (4, 8, 16):
            eps = 1e-3
            degree, _rel, _c = approx.min_degree(kappa, eps)
            coeffs, c, _rel2 = approx.approximate_inverse(kappa, degree)
            coeffs = coeffs * SAFETY
            c *= SAFETY
            phases, ares = solve_phases(lib, coeffs.tolist())

            A = random_hermitian(n_qubits, kappa, seed=1234 + 10 * n_qubits + kappa)
            block = qsvt_block(A, phases)
            fA = (block + block.conj().T) / 2  # real-part extraction (LCU, +1 ancilla)

            Ainv = np.linalg.inv(A)
            op_relerr = np.linalg.norm(fA - c * Ainv, 2) / np.linalg.norm(c * Ainv, 2)

            b = rng_b.standard_normal(2 ** n_qubits) + 1j * rng_b.standard_normal(2 ** n_qubits)
            b /= np.linalg.norm(b)
            x_exact = Ainv @ b
            x_exact /= np.linalg.norm(x_exact)
            xq = fA @ b
            success_prob = np.linalg.norm(xq) ** 2  # P(ancilla=0) proxy
            xq /= np.linalg.norm(xq)
            fidelity = abs(np.vdot(x_exact, xq))

            print(f"{n_qubits},{kappa},{degree},{eps:.0e},{op_relerr:.2e},"
                  f"{fidelity:.6f},{success_prob:.2e},{ares:.1e}")


if __name__ == "__main__":
    main()
