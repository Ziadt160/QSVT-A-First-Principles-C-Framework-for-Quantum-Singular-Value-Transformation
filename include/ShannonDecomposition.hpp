#pragma once
// Quantum Shannon Decomposition: compile an arbitrary n-qubit unitary into a
// native gate sequence (single-qubit gates + CNOTs).
//
// Recursion (split on the most-significant qubit):
//   U = (L1 (+) L2) . [[C,-S],[S,C]] . (R1 (+) R2)^H        (cosine-sine)
// The middle factor is a uniformly-controlled Ry on the top qubit; each
// block-diagonal multiplexor (V1 (+) V2) is demultiplexed via the eigendecomp
// V1 V2^H = E D^2 E^H into (I (x) E) . UCRz . (I (x) F), recursing the (n-1)-qubit
// factors E, F. Uniformly-controlled rotations expand into single-qubit
// rotations + CNOTs. Single-qubit unitaries are the base case.

#include <stdexcept>
#include <string>
#include <vector>

#include "Common.hpp"
#include "Gate.hpp"
#include "QrackTypes.hpp"

namespace qsvt {

class ShannonException : public std::runtime_error {
public:
    explicit ShannonException(const std::string& message)
        : std::runtime_error("Shannon Decomposition Error: " + message) {}
};

class ShannonDecomposition {
public:
    /// @param U A 2^n x 2^n unitary.
    explicit ShannonDecomposition(const Matrix& U);

    const std::vector<Gate>& gates() const { return gates_; }
    int numQubits() const { return n_; }

    /// Dense 2^n x 2^n unitary implemented by the gate list.
    Matrix circuitMatrix() const { return denseCircuit(gates_, n_); }

    /// Run the circuit on a Qrack register; logical qubit i maps to qubits[i].
    void applyTo(Qrack::QInterfacePtr qReg,
                 const std::vector<bitLenInt>& qubits) const
    {
        runCircuit(gates_, qReg, qubits);
    }

private:
    // Append gates implementing U on the given logical qubits (qubits.back() is
    // the most-significant / split qubit).
    void decompose(const Matrix& U, const std::vector<int>& qubits);

    // Append a 2-way multiplexor blockdiag(V1, V2) controlled by `msb`.
    void appendMultiplexor(const Matrix& V1, const Matrix& V2,
                           const std::vector<int>& lower, int msb);

    // Append a uniformly-controlled rotation (axis 'Y' or 'Z') on `target`,
    // controlled by `controls`, with 2^|controls| angles.
    void appendUCR(char axis, const RealVector& angles,
                   const std::vector<int>& controls, int target);

    std::vector<Gate> gates_;
    int n_;
};

} // namespace qsvt
