#pragma once
// The full Quantum Singular Value Transformation pipeline.
//
// Given a Hermitian operator A with ||A|| <= 1 and a phase sequence Phi (from
// QspAngleSolver), QSVT applies a polynomial transform to the eigenvalues of A:
// the top-left block of the QSVT operator equals P(A), where P is the same
// polynomial the QSP phases realise (Re P = the target function).
//
// We use the rotation-convention block-encoding
//     U_W = [[ A,            i*sqrt(I - A^2) ],
//            [ i*sqrt(I-A^2), A             ]]
// whose restriction to each eigenspace of A is exactly the QSP signal
// W(lambda) = [[lambda, i*s], [i*s, lambda]]. The QSVT operator interleaves U_W
// with projector-controlled phase rotations E(phi) = e^{i phi Z_anc}, mirroring
// the QSP product, so it reduces to QSP independently in every eigenspace.

#include <functional>
#include <vector>

#include "Common.hpp"
#include "Gate.hpp"

namespace qsvt {

/// Rotation-convention block-encoding U_W of a Hermitian contraction A.
/// @throws std::invalid_argument if A is not (numerically) Hermitian or ||A||>1.
Matrix rotationBlockEncoding(const Matrix& A);

/// Dense QSVT operator U_Phi = E(phi_0) * prod_{k=1..d} [ U_W * E(phi_k) ],
/// with E(phi) = e^{i phi Z_anc}. For a k x k A this is a 2k x 2k unitary
/// (one ancilla qubit on top of the system).
Matrix qsvtUnitary(const Matrix& A, const std::vector<double>& phases);

/// Top-left k x k block of the QSVT operator -- the realised transform, equal
/// to P(A) where P(lambda) = <0|U_qsp(lambda)|0> for the given phases.
Matrix qsvtBlock(const Matrix& A, const std::vector<double>& phases);

/// The QSVT operator as a native gate circuit on (1 + log2 k) qubits, with the
/// ancilla as the highest-index qubit. U_W is compiled via the Quantum Shannon
/// Decomposition; the projector rotations are single-qubit Z-rotations on the
/// ancilla.
std::vector<Gate> qsvtCircuit(const Matrix& A, const std::vector<double>& phases);

/// A compiled QSVT program: the matrix-function compiler turns (A, target f,
/// degree) into a runnable native-gate circuit plus its resource profile.
struct QsvtProgram {
    std::vector<double> phases; ///< QSP phases realising the target polynomial
    std::vector<Gate> circuit;  ///< native gate sequence (single-qubit + CNOT)
    int numQubits{0};           ///< system + 1 ancilla
    ResourceCounts resources;   ///< CNOT / single-qubit / total / depth
    double polyResidual{0.0};   ///< max |achieved poly - target| on [-1, 1]
    bool converged{false};      ///< whether the angle solver converged
};

/// Compile a degree-d polynomial approximation of `target(x)` applied to the
/// eigenvalues of Hermitian `A` into a QSVT circuit. `target` must have parity
/// d mod 2 and magnitude <= 1 on [-1, 1].
QsvtProgram compileMatrixFunction(const Matrix& A,
                                  const std::function<double(double)>& target,
                                  int degree);

} // namespace qsvt
