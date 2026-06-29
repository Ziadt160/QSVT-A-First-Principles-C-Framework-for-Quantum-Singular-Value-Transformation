"""Smoke test for the qsvt_native Python bindings."""
import numpy as np
import qsvt_native as q


def hermitian(eigs, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((len(eigs), len(eigs))) + 1j * rng.standard_normal((len(eigs), len(eigs)))
    U, _ = np.linalg.qr(a)
    return U @ np.diag(eigs) @ U.conj().T


def main():
    ok = True

    # 1. QSP angle finding for a degree-15 Chebyshev target.
    def cheb(n, x):
        if n == 0: return 1.0
        if n == 1: return x
        a, b = 1.0, x
        for _ in range(2, n + 1):
            a, b = b, 2 * x * b - a
        return b
    res = q.poly_to_angles(lambda x: 0.7 * cheb(15, x), 15)
    print(f"[QSP]   degree 15: residual={res['residual']:.2e}, converged={res['converged']}, "
          f"{len(res['phases'])} phases")
    ok &= res["converged"]

    # 2. Quantum Shannon Decomposition of a random 3-qubit unitary -> QASM.
    rng = np.random.default_rng(7)
    a = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    U, _ = np.linalg.qr(a)
    d = q.shannon_decompose(U)
    print(f"[QSD]   3-qubit: {d['cnot']} CNOTs, depth {d['depth']}, "
          f"recon_err={d['reconstruction_error']:.1e}, qasm {len(d['qasm'].splitlines())} lines")
    ok &= d["reconstruction_error"] < 1e-9

    # 3. Full QSVT: block(U_Phi) == P(A).
    A = hermitian([0.2, -0.5, 0.7, -0.1], 3)
    res2 = q.poly_to_angles(lambda x: 0.5 * cheb(3, x), 3)
    block = q.qsvt_block(A, res2["phases"])
    lam, V = np.linalg.eigh(A)
    pdiag = np.array([q.qsp_response(l, res2["phases"]) for l in lam])
    print(f"[QSVT]  block real part matches P(A): "
          f"{np.linalg.norm((block + block.conj().T)/2 - V @ np.diag(pdiag) @ V.conj().T):.1e}")

    # 4. Matrix-function compiler -> QASM.
    prog = q.compile_matrix_function(A, lambda x: 0.5 * cheb(3, x), 3)
    print(f"[COMPILE] {prog['num_qubits']} qubits, {prog['cnot']} CNOTs, qasm head: "
          f"{prog['qasm'].splitlines()[2]}")

    # 5. Hamiltonian simulation e^{-iHt}.
    H = hermitian([0.6, -0.4, 0.3, -0.7], 11)
    t = 2.5
    approx = q.hamiltonian_evolution(H, t, 21)
    exact = V @ np.diag(np.exp(-1j * t * lam)) @ V.conj().T  # wrong V; recompute
    lamH, VH = np.linalg.eigh(H)
    exact = VH @ np.diag(np.exp(-1j * t * lamH)) @ VH.conj().T
    err = np.linalg.norm(approx - exact)
    print(f"[HAMSIM] || e^-iHt_QSVT - exact || = {err:.2e}")
    ok &= err < 1e-2

    # 6. Spectral projector onto positive eigenspace.
    Hp = hermitian([-0.7, -0.4, 0.4, 0.8], 5)
    P = q.spectral_projector(Hp, 0.0, 0.1, 25)
    lamP, VP = np.linalg.eigh(Hp)
    exactP = VP @ np.diag([1.0 if l > 0 else 0.0 for l in lamP]) @ VP.conj().T
    print(f"[PROJ]  || P_QSVT - exact projector || = {np.linalg.norm(P - exactP):.2e}")

    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
