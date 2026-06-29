#pragma once
// Eigenvalue thresholding / spectral projection via QSVT (ground-state filtering,
// the third canonical QSVT application alongside inversion and simulation).
//
// To project onto the eigenspace of H with eigenvalues above a threshold mu, we
// apply an odd polynomial approximation of sign(x) to A = H - mu*I (assumed
// ||A|| <= 1, with eigenvalues bounded away from mu by a gap):
//
//   Pi_{>mu} = ( I + sign(H - mu) ) / 2.
//
// The smooth target is erf(x / w) (odd, |.| <= 1, ~ sign for |x| >> w). QSVT
// applies it; the Hermitian part of the block recovers sign(H - mu) exactly.

#include "Common.hpp"
#include "Gate.hpp"

namespace qsvt {

struct ThresholdProgram {
    std::vector<double> phases;
    std::vector<Gate> circuit;
    int numQubits{0};
    ResourceCounts resources;
    double fitResidual{0.0};
    bool converged{false};
    double mu{0.0}; // threshold
};

/// Compile the QSVT circuit applying sign(H - mu) (smoothed over width `w`).
/// `degree` sets the polynomial truncation; ~1/w sets the achievable sharpness.
ThresholdProgram compileEigenvalueThreshold(const Matrix& H, double mu, double w,
                                            int degree);

/// Dense spectral projector Pi_{>mu} ~ (I + sign(H - mu)) / 2 from the QSVT block.
Matrix spectralProjector(const Matrix& H, const ThresholdProgram& program);

} // namespace qsvt
