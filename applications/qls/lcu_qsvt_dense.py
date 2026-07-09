"""Capstone, dense validation: QSVT on top of the LCU (sparse) block-encoding.

W2 validated QSVT with the DENSE block-encoding. This validates the whole SPARSE
pipeline end to end, densely (no Qrack): a local Hamiltonian A = Σ c_k P_k is
LCU-block-encoded into a unitary U, and the qubitization-QSVT sequence with
sym_qsp angles applies a target polynomial f to A/α.

Key fact: for Hermitian A with real coefficients, U = PREP† SELECT PREP is
Hermitian AND U² = I (SELECT² = I since Paulis square to I, PREP is orthogonal).
That is exactly the qubitization form, so the standard QSVT reflection sequence
    U_Φ = R(φ_0) · Π_k [ U · R(φ_k) ],   R(φ) = e^{i φ (2Π - I)},  Π = |0..0>_a<0..0|
gives  <0..0|_a U_Φ |0..0>_a = P(A/α),  Re P = f  (same convention as W2).

Angles come from THIS project's sym_qsp solver via its zero-dependency C ABI.

    python lcu_qsvt_dense.py
"""

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "..", "packages", "qsp-angles"))

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.array([[1, 0], [0, -1]], complex),
}


def pauli_string(s):
    m = np.array([[1]], dtype=complex)
    for ch in s:
        m = np.kron(m, PAULI[ch])
    return m


def ensure_solver(lib_path="/tmp/libqsp_angles_c.so"):
    if not os.path.exists(lib_path):
        subprocess.run(
            ["g++", "-O3", "-fPIC", "-shared", "-std=c++17",
             "-I" + os.path.join(PKG, "capi"), "-I" + os.path.join(PKG, "src"),
             os.path.join(PKG, "capi", "qsp_angles.cpp"),
             os.path.join(PKG, "src", "SymQspAngleSolver.cpp"),
             "-o", lib_path], check=True)
    lib = ctypes.CDLL(lib_path)
    D, PD, PI = ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.qsp_solve_chebyshev.restype = ctypes.c_int
    lib.qsp_solve_chebyshev.argtypes = [ctypes.c_int, PD, ctypes.c_int, PD, ctypes.c_int, PI, PD, PI]
    return lib


def solve_phases(lib, cheb_coeffs):
    n = len(cheb_coeffs)
    c = (ctypes.c_double * n)(*cheb_coeffs)
    ph = (ctypes.c_double * n)()
    ln, res, cv = ctypes.c_int(0), ctypes.c_double(0), ctypes.c_int(0)
    st = lib.qsp_solve_chebyshev(n - 1, c, n, ph, n, ctypes.byref(ln), ctypes.byref(res), ctypes.byref(cv))
    if st != 0:
        raise RuntimeError(f"solve status {st}")
    return np.array(ph[: ln.value]), res.value


def lcu_full_unitary(n, terms):
    """Full LCU block-encoding unitary U on (m ancilla + n system) qubits, and alpha.
    Ancilla are the HIGH qubits, so |0..0>_a is the top-left 2^n block."""
    L = len(terms)
    m = int(np.ceil(np.log2(L)))
    M = 1 << m
    coeffs = np.array([c for c, _ in terms], complex)
    alpha = float(np.sum(np.abs(coeffs)))
    psi = np.zeros(M, complex)
    psi[:L] = np.sqrt(np.abs(coeffs) / alpha)
    rng = np.random.default_rng(0)
    PREP, _ = np.linalg.qr(np.column_stack([psi, rng.standard_normal((M, M - 1)) + 0j]))
    dimn = 1 << n
    SEL = np.zeros((M * dimn, M * dimn), complex)
    for k in range(M):
        if k < L:
            c, s = terms[k]
            Uk = (c / abs(c)) * pauli_string(s)
        else:
            Uk = np.eye(dimn, dtype=complex)
        SEL[k * dimn:(k + 1) * dimn, k * dimn:(k + 1) * dimn] = Uk
    PREPn = np.kron(PREP, np.eye(dimn, dtype=complex))
    U = PREPn.conj().T @ SEL @ PREPn
    return U, alpha, m


def qsvt_block(U, phases, n_system, m):
    """<0..0|_a U_Phi |0..0>_a with U_Phi = R(phi0) prod[W R(phik)].

    The block-encoding U is Hermitian and U^2 = I (a reflection). The
    qubitization WALK operator W = U (2Pi - I) has a genuine 2D rotation action
    (Ry(2 arccos lambda)) on each eigenvalue subspace, so the standard QSP/QSVT
    sequence with R(phi) = e^{i phi (2Pi - I)} and this project's Wx-convention
    sym_qsp angles applies the target polynomial. (Using U directly -- a
    reflection, trace 0 -- gives the wrong polynomial; the walk operator is the
    fix. Verified: Re-part matches f(A/alpha) to ~1e-16.)
    """
    dim = U.shape[0]
    dimn = 1 << n_system
    z = -np.ones(dim)  # diag(2Pi - I): +1 on the ancilla-|0..0> block, -1 else
    z[:dimn] = 1.0
    W = U @ np.diag(z)  # qubitization walk operator

    def R(phi):
        return np.diag(np.exp(1j * phi * z))

    Uphi = R(phases[0])
    for p in phases[1:]:
        Uphi = Uphi @ W @ R(p)
    return Uphi[:dimn, :dimn]


def main():
    lib = ensure_solver()

    # Small Hermitian test operator as a sum of Paulis (2 system qubits),
    # normalised so ||A|| < 1 is not required here (block encodes A/alpha).
    terms = [(0.5, "XX"), (0.3, "ZI"), (0.2, "IZ"), (0.4, "XZ")]
    n = 2
    A = sum(c * pauli_string(s) for c, s in terms)
    U, alpha, m = lcu_full_unitary(n, terms)

    herm_err = np.linalg.norm(U - U.conj().T)
    invol_err = np.linalg.norm(U @ U - np.eye(U.shape[0]))
    block0 = U[: 1 << n, : 1 << n]
    be_err = np.linalg.norm(block0 - A / alpha)
    print(f"LCU U: Hermitian err={herm_err:.1e}, U^2=I err={invol_err:.1e}, "
          f"block==A/alpha err={be_err:.1e}  (alpha={alpha:.2f}, m={m} ancilla)")

    # Target f = 0.7 * T_3 (odd), applied to A/alpha via QSVT.
    d = 3
    coeffs = [0.0] * (d + 1)
    coeffs[d] = 0.7
    phases, ares = solve_phases(lib, coeffs)

    block = qsvt_block(U, phases, n, m)
    fA = (block + block.conj().T) / 2  # Hermitian (real-part) component

    # Reference: f(A/alpha) = 0.7 * T_3(A/alpha) via eigen-decomposition.
    w, Vv = np.linalg.eigh(A / alpha)
    fref = Vv @ np.diag(0.7 * np.cos(3 * np.arccos(np.clip(w, -1, 1)))) @ Vv.conj().T

    err = np.linalg.norm(fA - fref) / np.linalg.norm(fref)
    print(f"sym_qsp angles: {len(phases)} phases, residual={ares:.1e}")
    print(f"QSVT-on-LCU block (Hermitian part) vs 0.7*T_3(A/alpha): rel err = {err:.2e}")
    print("PASS" if err < 1e-9 else "MISMATCH (convention?)")


if __name__ == "__main__":
    main()
