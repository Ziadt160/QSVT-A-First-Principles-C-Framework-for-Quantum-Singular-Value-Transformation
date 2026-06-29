#pragma once
// Compile a two-qubit unitary into a native gate sequence (single-qubit gates +
// CNOTs) using the KAK decomposition, then run it on the Qrack simulator.
//
// Pipeline: U = K1 * A * K2 (KAK), where K1, K2 are local (tensor products of
// single-qubit gates) and A is the canonical entangler exp(i(ax XX + ay YY +
// az ZZ)) up to a global phase. The locals are split into their single-qubit
// factors (Van Loan-Pitsianis), and the entangler is realised with the exact
// identity exp(i t ZZ) = CNOT (I (x) Rz(-2t)) CNOT, conjugated into XX / YY by
// Hadamard / S gates. The result is correct, though not gate-count optimal
// (an optimal canonical circuit uses only three CNOTs).

#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "Common.hpp"
#include "Gate.hpp"
#include "QrackTypes.hpp"

namespace qsvt {

class TwoQubitSynthesis {
public:
    /// @param U The two-qubit unitary to compile (computational basis, qubit 0
    ///          is the low/least-significant qubit).
    explicit TwoQubitSynthesis(const Eigen::Matrix4cd& U);

    /// The synthesised gate sequence, in application order.
    const std::vector<Gate>& gates() const { return gates_; }

    /// Dense 4x4 unitary implemented by the gate list (Eigen reference model).
    Eigen::Matrix4cd circuitMatrix() const;

    /// Apply the circuit to a Qrack register. @p q0 is the low qubit, @p q1 the
    /// high qubit; they must be distinct.
    void applyTo(Qrack::QInterfacePtr qReg, bitLenInt q0, bitLenInt q1) const;

    /// Split K = A (x) B into single-qubit factors (A on the high qubit, B on
    /// the low qubit). Exposed for testing.
    static std::pair<Eigen::Matrix2cd, Eigen::Matrix2cd>
    factorizeKron(const Eigen::Matrix4cd& K);

private:
    void build();

    Eigen::Matrix4cd U_;
    std::vector<Gate> gates_;
};

/// Synthesize a two-qubit unitary into native gates acting on logical qubits
/// @p qLow (least significant) and @p qHigh. Uses KAK plus a 4-CNOT canonical
/// entangler (the XX and ZZ rotations share a CNOT pair). Shared by
/// TwoQubitSynthesis and the n=2 base case of ShannonDecomposition.
std::vector<Gate> synthesizeTwoQubit(const Eigen::Matrix4cd& U, int qLow,
                                     int qHigh);

} // namespace qsvt
