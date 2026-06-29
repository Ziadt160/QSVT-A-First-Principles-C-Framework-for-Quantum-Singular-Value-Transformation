#include "BlockEncoding.hpp"

#include <cmath>

#include <eigen3/Eigen/Eigenvalues>

#include "ShannonDecomposition.hpp"

namespace qsvt {

namespace {

/// Principal square root of a Hermitian positive-semidefinite matrix.
Matrix hermitianSqrt(const Matrix& h)
{
    Eigen::SelfAdjointEigenSolver<Matrix> solver(h);
    if (solver.info() != Eigen::Success) {
        throw BlockEncodingException("matrix square root failed to converge");
    }
    // Clamp tiny negative eigenvalues (round-off) to zero before the sqrt.
    RealVector roots = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    return solver.eigenvectors() * roots.cast<Complex>().asDiagonal() *
           solver.eigenvectors().adjoint();
}

} // namespace

BlockEncoding::BlockEncoding(Qrack::QInterfacePtr qReg, double scalar)
    : qReg_(std::move(qReg))
{
    if (std::abs(scalar) > 1.0) {
        throw BlockEncodingException("scalar must lie in [-1, 1]");
    }

    const double p = scalar;
    const double q = std::sqrt(1.0 - scalar * scalar);

    Matrix m(2, 2);
    m(0, 0) = Complex(p, 0);
    m(0, 1) = Complex(q, 0);
    m(1, 0) = Complex(q, 0);
    m(1, 1) = Complex(-p, 0);

    unitary_ = toQMatrix(m);
}

BlockEncoding::BlockEncoding(Qrack::QInterfacePtr qReg, const Matrix& matrix)
    : qReg_(std::move(qReg))
{
    if (matrix.rows() != matrix.cols()) {
        throw BlockEncodingException("matrix must be square");
    }

    const Eigen::Index n = matrix.rows();

    // The spectral norm (largest singular value) must be a contraction.
    Eigen::JacobiSVD<Matrix> svd(matrix);
    if (svd.singularValues().size() > 0 &&
        svd.singularValues()(0) > 1.0 + kTolerance) {
        throw BlockEncodingException("matrix spectral norm is larger than 1");
    }

    const Matrix I = Matrix::Identity(n, n);
    const Matrix dimTop    = hermitianSqrt(I - matrix * matrix.adjoint());
    const Matrix dimBottom = hermitianSqrt(I - matrix.adjoint() * matrix);

    // U = [[ A,            sqrt(I - A A^H) ],
    //      [ sqrt(I-A^HA), -A^H            ]]   is unitary for any contraction A.
    Matrix u(2 * n, 2 * n);
    u.topLeftCorner(n, n)     = matrix;
    u.topRightCorner(n, n)    = dimTop;
    u.bottomLeftCorner(n, n)  = dimBottom;
    u.bottomRightCorner(n, n) = -matrix.adjoint();

    unitary_ = toQMatrix(u);
}

void BlockEncoding::ensureSingleQubit(const char* op) const
{
    if (unitary_.rows() != 2) {
        throw BlockEncodingException(
            std::string(op) +
            ": multi-qubit block-encodings must be compiled to native gates "
            "(see KAK / CS decomposition); only the single-qubit case is wired "
            "to the simulator.");
    }
}

void BlockEncoding::apply(bitLenInt target) const
{
    ensureSingleQubit("apply");
    qReg_->Mtrx(unitary_.data(), target);
}

void BlockEncoding::apply(const std::vector<bitLenInt>& qubits) const
{
    if ((Eigen::Index(1) << qubits.size()) != unitary_.rows()) {
        throw BlockEncodingException(
            "apply: qubit count must equal log2(dimension)");
    }

    // Widen the stored (possibly single-precision) unitary back to double and
    // compile it to native gates.
    Matrix u(unitary_.rows(), unitary_.cols());
    for (Eigen::Index r = 0; r < unitary_.rows(); ++r) {
        for (Eigen::Index c = 0; c < unitary_.cols(); ++c) {
            u(r, c) = Complex(std::real(unitary_(r, c)), std::imag(unitary_(r, c)));
        }
    }

    ShannonDecomposition synth(u);
    synth.applyTo(qReg_, qubits);
}

void BlockEncoding::apply_adjoint(bitLenInt target) const
{
    ensureSingleQubit("apply_adjoint");
    qReg_->Mtrx(unitary_.adjoint().eval().data(), target);
}

void BlockEncoding::controlled_apply(const std::vector<bitLenInt>& controls,
                                     bitLenInt target) const
{
    ensureSingleQubit("controlled_apply");
    qReg_->MCMtrx(controls, unitary_.data(), target);
}

void BlockEncoding::anti_controlled_apply(const std::vector<bitLenInt>& controls,
                                          bitLenInt target) const
{
    ensureSingleQubit("anti_controlled_apply");
    qReg_->MACMtrx(controls, unitary_.data(), target);
}

} // namespace qsvt
