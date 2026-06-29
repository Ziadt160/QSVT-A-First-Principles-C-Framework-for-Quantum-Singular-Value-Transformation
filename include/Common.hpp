#pragma once
// Shared linear-algebra types for the QSVT framework.
//
// Everything that is "pure math" (decompositions, LCU, etc.) works in
// double-precision Eigen and lives in the `qsvt` namespace. The classes that
// talk to the Qrack simulator additionally include QrackTypes.hpp, which bridges
// these double-precision matrices into Qrack's (possibly single-precision)
// complex layout.

#include <complex>
#include <eigen3/Eigen/Dense>

namespace qsvt {

/// Scalar type used throughout the math layer.
using Complex = std::complex<double>;

/// Dynamically-sized, double-precision complex matrix/vector.
using Matrix     = Eigen::MatrixXcd;
using Vector     = Eigen::VectorXcd;
using RealVector = Eigen::VectorXd;

/// Default numerical tolerance for rank/zero comparisons in the decompositions.
inline constexpr double kTolerance = 1e-10;

} // namespace qsvt
