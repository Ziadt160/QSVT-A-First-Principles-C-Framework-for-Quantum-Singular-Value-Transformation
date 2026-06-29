#pragma once
// Block-encoding of a scalar or matrix into a unitary acting on the simulator.

#include <stdexcept>
#include <string>
#include <vector>

#include "Common.hpp"
#include "QrackTypes.hpp"

namespace qsvt {

class BlockEncodingException : public std::runtime_error {
public:
    explicit BlockEncodingException(const std::string& message)
        : std::runtime_error("Block Encoding Error: " + message) {}
};

/// Builds a unitary U whose top-left block is the target operator, so that the
/// operator can be applied to a subspace of the simulator's state.
///
/// Scalar `a` in [-1, 1] is encoded as the reflection [[a, q], [q, -a]] with
/// q = sqrt(1 - a^2). A contraction `A` (||A|| <= 1) is encoded via the standard
/// dilation U = [[A, sqrt(I - A A^H)], [sqrt(I - A^H A), -A^H]].
class BlockEncoding {
public:
    /// @param qReg   The quantum simulator interface.
    /// @param scalar Scalar to encode; must satisfy |scalar| <= 1.
    BlockEncoding(Qrack::QInterfacePtr qReg, double scalar);

    /// @param qReg   The quantum simulator interface.
    /// @param matrix Operator to encode; its spectral norm must be <= 1.
    BlockEncoding(Qrack::QInterfacePtr qReg, const Matrix& matrix);

    /// Apply the block-encoding unitary U to a target qubit (single-qubit
    /// encodings only).
    void apply(bitLenInt target) const;

    /// Apply a multi-qubit block-encoding by compiling its unitary to native
    /// gates (Quantum Shannon Decomposition) and running them. @p qubits maps
    /// logical qubit i (LSB first) to a physical qubit; its length must be
    /// log2(dimension()).
    void apply(const std::vector<bitLenInt>& qubits) const;

    /// Apply the adjoint (inverse) of the block-encoding unitary.
    void apply_adjoint(bitLenInt target) const;

    /// Apply the controlled unitary C-U.
    void controlled_apply(const std::vector<bitLenInt>& controls, bitLenInt target) const;

    /// Apply the anti-controlled unitary.
    void anti_controlled_apply(const std::vector<bitLenInt>& controls, bitLenInt target) const;

    /// The full block-encoding unitary (Qrack layout).
    const QMatrix& unitary() const { return unitary_; }

    /// Dimension of the block-encoding unitary (2 for a scalar).
    Eigen::Index dimension() const { return unitary_.rows(); }

private:
    void ensureSingleQubit(const char* op) const;

    Qrack::QInterfacePtr qReg_;
    QMatrix unitary_;
};

} // namespace qsvt
