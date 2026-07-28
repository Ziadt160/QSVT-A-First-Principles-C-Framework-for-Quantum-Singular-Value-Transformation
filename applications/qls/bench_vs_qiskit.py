"""Honest head-to-head vs Qiskit (qiskit.synthesis.qs_decomposition).

Two comparisons:
  (A) generic n-qubit unitary synthesis -> CNOT count + time. Qiskit's QSD is
      near-optimal here; our from-scratch synthesizer is ~1.4x behind. Reported
      straight -- NOT a lead.
  (B) block-encoding a LOCAL Hamiltonian. Our sparse LCU uses poly(n) gates; a
      Qiskit user with no sparse-block-encoding primitive would synthesize the
      dense dilation unitary with qs_decomposition -> ~0.48*4^n gates. This is
      where the real, defensible win is.

    python bench_vs_qiskit.py
"""

import time

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import qs_decomposition

I2 = np.eye(2, dtype=complex)
PAULI = {"I": I2, "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]], complex),
         "Z": np.array([[1, 0], [0, -1]], complex)}


def pauli_string(s):
    m = np.array([[1]], dtype=complex)
    for ch in s:
        m = np.kron(m, PAULI[ch])
    return m


def cx_count(circ):
    t = transpile(circ, basis_gates=["cx", "u"], optimization_level=0)
    return t.count_ops().get("cx", 0)


def qsd_cx_and_time(U, reps=3):
    ts = []
    circ = None
    for _ in range(reps):
        t0 = time.perf_counter()
        circ = qs_decomposition(U)
        ts.append(time.perf_counter() - t0)
    return cx_count(circ), float(np.median(ts)) * 1e3


def tfim(n, J=1.0, h=0.6):
    terms = []
    for i in range(n - 1):
        s = ["I"] * n; s[i] = s[i + 1] = "Z"; terms.append((J, "".join(s)))
    for i in range(n):
        s = ["I"] * n; s[i] = "X"; terms.append((h, "".join(s)))
    return terms


def rotation_block_encoding(A):
    """U_W = [[A, iB],[iB, A]], B = sqrt(I - A^2), for a Hermitian contraction A."""
    w, V = np.linalg.eigh(A)
    B = (V * np.sqrt(np.clip(1 - w ** 2, 0, None))) @ V.conj().T
    return np.block([[A, 1j * B], [1j * B, A]])


def main():
    print("=== (A) generic unitary synthesis: ours (Shannon) vs Qiskit QSD ===")
    # Our framework's Shannon CNOT counts (from src/, confirmed by resource_scaling):
    ours_shannon = {2: 4, 3: 28, 4: 136, 5: 592}
    print(f"{'n':>3} {'ours_cx':>8} {'qiskit_cx':>10} {'qiskit_ms':>10} {'ratio(ours/qk)':>15}")
    for n in (2, 3, 4, 5):
        U = np.array(random_unitary(2 ** n, seed=7 + n))
        qk_cx, qk_ms = qsd_cx_and_time(U)
        o = ours_shannon[n]
        print(f"{n:>3} {o:>8} {qk_cx:>10} {qk_ms:>10.1f} {o / max(qk_cx,1):>15.2f}")

    print("\n=== (B) local-Hamiltonian block-encoding: our LCU (poly n) vs Qiskit dense QSD ===")
    # Our LCU CNOT counts (from applications/qls/lcu_be_validate.cpp, TFIM):
    ours_lcu = {2: 42, 3: 126, 4: 172, 5: 324, 6: 390}
    print(f"{'n':>3} {'ours_LCU_cx':>12} {'qiskit_dense_cx':>16} {'speedup(qk/ours)':>17}")
    for n in (2, 3, 4, 5):
        A = sum(c * pauli_string(s) for c, s in tfim(n))
        A = A / np.max(np.abs(np.linalg.eigvalsh(A)))   # normalise ||A|| <= 1
        Uw = rotation_block_encoding(A)                 # 2^(n+1) dilation unitary
        qk_cx, _ = qsd_cx_and_time(Uw, reps=1)
        o = ours_lcu[n]
        print(f"{n:>3} {o:>12} {qk_cx:>16} {qk_cx / o:>16.1f}x")


if __name__ == "__main__":
    main()
