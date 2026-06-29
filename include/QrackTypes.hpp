#pragma once
// Bridges the double-precision math layer (Common.hpp) to Qrack's complex type.
//
// Qrack may be compiled with single- or double-precision floats (Qrack::real1),
// so we never assume the two complex types are identical and always convert
// explicitly.

#include <qrack/qfactory.hpp>
#include <eigen3/Eigen/Dense>

#include "Common.hpp"

namespace qsvt {

/// A matrix laid out exactly the way Qrack's gate functions expect the raw
/// buffer: row-major, so `.data()` yields [m00, m01, m10, m11, ...].
using QMatrix = Eigen::Matrix<Qrack::complex, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>;

/// Convert a double-precision Eigen matrix into Qrack's complex layout.
inline QMatrix toQMatrix(const Matrix& m)
{
    QMatrix out(m.rows(), m.cols());
    for (Eigen::Index r = 0; r < m.rows(); ++r) {
        for (Eigen::Index c = 0; c < m.cols(); ++c) {
            out(r, c) = Qrack::complex(static_cast<Qrack::real1>(m(r, c).real()),
                                       static_cast<Qrack::real1>(m(r, c).imag()));
        }
    }
    return out;
}

} // namespace qsvt
