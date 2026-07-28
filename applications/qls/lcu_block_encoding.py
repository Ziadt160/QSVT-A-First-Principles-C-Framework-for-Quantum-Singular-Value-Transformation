"""Phase-2 spike: LCU (PREPARE+SELECT) block-encoding of a sparse/local
Hamiltonian, validated densely. This is the construction that breaks the
gate-count wall of resource_scaling.py: for A = sum_k c_k P_k (L Pauli terms),
the block-encoding costs O(L * poly(n)) -- poly(n) for a local model -- instead
of the ~0.58*4^n of a dense Shannon decomposition.

Construction (Gilyen-Su-Low-Wiebe / Childs):
  PREPARE |0>_a = sum_k sqrt(|c_k|/alpha) |k>,   alpha = sum_k |c_k|
  SELECT       = sum_k |k><k|_a (x) U_k,         U_k = (c_k/|c_k|) P_k
  U = (PREPARE† (x) I) SELECT (PREPARE (x) I)
  =>  <0|_a U |0>_a = sum_k (c_k/alpha) P_k = A / alpha   (subnormalization alpha)

We build U densely and check the top-left block == A/alpha to machine precision,
then contrast the gate-count SCALING (poly(n) vs 4^n) against the dense path.

    python lcu_block_encoding.py
"""

import numpy as np

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


def tfim_terms(n, J=1.0, h=0.6):
    """1D transverse-field Ising (open chain): H = J Σ Z_i Z_{i+1} + h Σ X_i."""
    terms = []
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = s[i + 1] = "Z"
        terms.append((J, "".join(s)))
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        terms.append((h, "".join(s)))
    return terms


def lcu_block_encoding(n, terms):
    """Build the dense LCU block-encoding U and return (top-left block, alpha)."""
    L = len(terms)
    m = int(np.ceil(np.log2(L)))
    M = 1 << m
    coeffs = np.array([c for c, _ in terms], dtype=complex)
    alpha = float(np.sum(np.abs(coeffs)))

    # PREPARE: any unitary with first column = sqrt(|c_k|/alpha) (phase is
    # irrelevant -- it cancels in PREP† ... PREP).
    psi = np.zeros(M, complex)
    psi[:L] = np.sqrt(np.abs(coeffs) / alpha)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(np.column_stack([psi, rng.standard_normal((M, M - 1)) + 0j]))
    PREP = Q

    # SELECT = blockdiag over k of U_k = (c_k/|c_k|) P_k (padding blocks = I).
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
    return U[:dimn, :dimn], alpha


def lcu_cnot_estimate(n, terms):
    """A documented gate-count MODEL (scaling is the point, not the constant).

    PREPARE ~ Mottonen state prep on m ancilla qubits (~2^{m+1} CNOTs).
    SELECT  ~ L terms, each an m-controlled Pauli string of weight w_k
              (~2m + w_k CNOTs per term with a borrowed ancilla).
    """
    L = len(terms)
    m = int(np.ceil(np.log2(L)))
    prep = (1 << (m + 1))
    select = sum(2 * m + sum(ch != "I" for ch in s) for _c, s in terms)
    return prep + select


def dense_cnot(n):
    """The framework's dense block-encoding law (confirmed by resource_scaling.cpp)."""
    return 0.58 * (4 ** (n + 1))


def main():
    print(f"{'n':>3} {'L_terms':>8} {'anc':>4} {'block_err':>12} "
          f"{'LCU_cnots':>10} {'dense_cnots':>13} {'speedup':>10}")
    for n in range(2, 11):
        terms = tfim_terms(n)
        alpha = sum(abs(c) for c, _ in terms)
        L = len(terms)
        m = int(np.ceil(np.log2(L)))

        block_err = float("nan")
        if n <= 8:  # dense check only where 2^(n+log2 L) is affordable
            block, alpha2 = lcu_block_encoding(n, terms)
            A = sum(c * pauli_string(s) for c, s in terms)
            block_err = np.linalg.norm(block - A / alpha2) / np.linalg.norm(A / alpha2)

        lcu = lcu_cnot_estimate(n, terms)
        dense = dense_cnot(n)
        print(f"{n:>3} {L:>8} {m:>4} {block_err:>12.2e} "
              f"{lcu:>10} {dense:>13.3e} {dense/lcu:>10.1f}x")


if __name__ == "__main__":
    main()
