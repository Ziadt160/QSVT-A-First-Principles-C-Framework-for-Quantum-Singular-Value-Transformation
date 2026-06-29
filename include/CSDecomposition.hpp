#pragma once
// Cosine-Sine decomposition (CSD) of an even-dimensional unitary.
//
// Given a 2n x 2n unitary U split into four n x n blocks, the CSD writes
//
//     U = [ L1  0 ] [  C  -S ] [ R1  0 ]^H
//         [ 0  L2 ] [  S   C ] [ 0  R2 ]
//
// where L1, L2, R1, R2 are n x n unitaries and C = diag(cos theta),
// S = diag(sin theta). This is the matrix-factorization workhorse behind
// recursive multi-qubit gate synthesis (quantum Shannon decomposition): the
// cos/sin middle factor is a uniformly-controlled (multiplexed) rotation, and
// the block-diagonal factors recurse on half the dimension. Unlike the 4x4-only
// KAK, it applies to any even matrix size.

#include <stdexcept>
#include <string>

#include "Common.hpp"

namespace qsvt {

class CSDecompositionException : public std::runtime_error {
public:
    explicit CSDecompositionException(const std::string& message)
        : std::runtime_error("CS Decomposition Error: " + message) {}
};

class CSDecomposition {
public:
    struct Result {
        Matrix     L1, L2;  // left unitaries (n x n)
        Matrix     R1, R2;  // right unitaries (n x n); used as R1^H, R2^H
        RealVector theta;   // n rotation angles; C = cos(theta), S = sin(theta)
    };

    /// @param U A unitary matrix of even dimension 2n.
    explicit CSDecomposition(const Matrix& U);

    /// Compute the decomposition.
    Result solve();

    /// Rebuild the full 2n x 2n unitary from a Result (handy for tests/use).
    static Matrix reconstruct(const Result& r);

private:
    Matrix U_;
    Eigen::Index n_; // half dimension
};

} // namespace qsvt
