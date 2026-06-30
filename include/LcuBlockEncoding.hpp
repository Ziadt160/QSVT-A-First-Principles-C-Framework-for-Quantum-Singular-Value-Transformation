#pragma once
// LCU (PREPARE + SELECT) block-encoding, synthesised to native gates.
//
// For a Hamiltonian / operator written as a Linear Combination of Unitaries
//     A = sum_k c_k P_k        (P_k a Pauli string, c_k a complex coefficient)
// this builds a unitary U on (system + ancilla) qubits such that the sub-block
// of U on the subspace where ALL ancilla qubits are |0> equals A / alpha, with
//     alpha = sum_k |c_k|   (the subnormalisation).
//
// Construction (Gilyen-Su-Low-Wiebe / Childs):
//     PREPARE |0>_a = sum_k sqrt(|c_k|/alpha) |k>
//     SELECT        = sum_k |k><k|_a (x) U_k,   U_k = (c_k/|c_k|) P_k
//     U = (PREPARE^dagger (x) I) SELECT (PREPARE (x) I)
//   =>  <0|_a U |0>_a = sum_k (c_k/alpha) P_k = A / alpha
//
// The construction matches the validated dense Python reference
// applications/qls/lcu_block_encoding.py to machine precision.
//
// Qubit layout (documented):
//   - system qubits  : 0 .. n_system-1          (qubit 0 = least significant)
//   - ancilla qubits : n_system .. n_system+m-1  (the HIGH qubits)
// With the ancilla as the high qubits, the all-ancilla-|0> block is the
// top-left 2^n_system x 2^n_system block of denseCircuit(gates, n_system+m).
//
// Pauli-string indexing: a term's `pauli` string has its FIRST character as the
// most-significant system qubit, matching the Python kron convention
//   pauli_string(s) = s[0] (x) s[1] (x) ... (x) s[L-1].
// So character at string position i acts on system qubit (n_system - 1 - i).

#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "Common.hpp"
#include "Gate.hpp"

namespace qsvt {

/// One LCU term: coefficient c_k and a Pauli string (e.g. "IXZI", length =
/// n_system). The string's first character is the most-significant system qubit.
struct PauliTerm {
    Complex coeff;
    std::string pauli;
};

class LcuBlockEncoding {
public:
    /// Build the block-encoding for A = sum_k terms[k].coeff * P(terms[k].pauli)
    /// on `n_system` system qubits. Each term's pauli string must have length
    /// n_system. Coefficients with |c_k| == 0 are rejected.
    LcuBlockEncoding(const std::vector<PauliTerm>& terms, int n_system);

    /// The native-gate circuit on (n_system + numAncilla()) qubits.
    /// {Gate::single, Gate::cnot} only.
    const std::vector<Gate>& gates() const { return gates_; }

    int numAncilla() const { return n_ancilla_; }
    int numSystem() const { return n_system_; }

    /// alpha = sum_k |c_k|, the subnormalisation: the all-ancilla-|0> block of
    /// the circuit equals A / alpha.
    double alpha() const { return alpha_; }

    /// Number of LCU terms L.
    int numTerms() const { return static_cast<int>(terms_.size()); }

private:
    std::vector<PauliTerm> terms_;
    int n_system_{0};
    int n_ancilla_{0};
    double alpha_{0.0};
    std::vector<Gate> gates_;
};

} // namespace qsvt
