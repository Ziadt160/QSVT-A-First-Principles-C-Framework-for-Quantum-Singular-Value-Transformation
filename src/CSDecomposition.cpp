#include "CSDecomposition.hpp"

#include <cmath>
#include <vector>

#include <eigen3/Eigen/SVD>

namespace qsvt {

CSDecomposition::CSDecomposition(const Matrix& U)
    : U_(U)
{
    if (U_.rows() != U_.cols()) {
        throw CSDecompositionException("matrix must be square");
    }
    if (U_.rows() % 2 != 0 || U_.rows() == 0) {
        throw CSDecompositionException("matrix dimension must be even and positive");
    }
    n_ = U_.rows() / 2;
}

CSDecomposition::Result CSDecomposition::solve()
{
    const Eigen::Index n = n_;

    // Partition U into n x n blocks.
    const Matrix A  = U_.topLeftCorner(n, n);
    const Matrix B  = U_.topRightCorner(n, n);
    const Matrix Cb = U_.bottomLeftCorner(n, n);
    const Matrix D  = U_.bottomRightCorner(n, n);

    // SVD of the top-left block: A = L1 * C * R1^H, with cosines C = singular
    // values (descending, all in [0, 1] because U is unitary). BDCSVD
    // (divide-and-conquer) is markedly faster than one-sided Jacobi for the
    // larger blocks in the recursion.
    Eigen::BDCSVD<Matrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success) {
        throw CSDecompositionException("SVD of the top-left block failed");
    }

    Result r;
    r.L1 = svd.matrixU();
    r.R1 = svd.matrixV();

    const RealVector cosv = svd.singularValues().cwiseMin(1.0).cwiseMax(0.0);

    // L2 from the bottom-left block: Cb * R1 = L2 * S, so column k of (Cb R1)
    // equals sin(theta_k) * (L2 column k). We read the sine straight off the
    // column norm rather than from acos(cos): since (Cb R1)^H (Cb R1) = I - C^2,
    // ||(Cb R1) col k|| == sin(theta_k) exactly. This is robust when A is itself
    // (near-)unitary, where acos(1 - eps) would otherwise report a spurious
    // nonzero sine for a column that is actually zero.
    const Matrix T = Cb * r.R1;
    RealVector sinv(n);
    for (Eigen::Index k = 0; k < n; ++k) {
        sinv(k) = T.col(k).norm();
    }
    r.theta = RealVector(n);
    for (Eigen::Index k = 0; k < n; ++k) {
        r.theta(k) = std::atan2(sinv(k), cosv(k)); // theta in [0, pi/2]
    }

    // Columns with sin ~ 0 are free and are filled by orthonormal completion.
    r.L2 = Matrix::Zero(n, n);

    std::vector<Eigen::Index> definedCols;
    std::vector<Eigen::Index> freeCols;
    for (Eigen::Index k = 0; k < n; ++k) {
        if (sinv(k) > kTolerance) {
            r.L2.col(k) = T.col(k) / sinv(k);
            definedCols.push_back(k);
        } else {
            freeCols.push_back(k);
        }
    }

    if (!freeCols.empty()) {
        // The defined columns are orthonormal (T^H T = S^2). Build an
        // orthonormal basis of their complement and assign it to the free
        // columns so L2 is unitary.
        Matrix complement;
        if (definedCols.empty()) {
            complement = Matrix::Identity(n, n); // any orthonormal basis works
        } else {
            Matrix defined(n, static_cast<Eigen::Index>(definedCols.size()));
            for (std::size_t j = 0; j < definedCols.size(); ++j) {
                defined.col(static_cast<Eigen::Index>(j)) = r.L2.col(definedCols[j]);
            }
            // Left-singular vectors beyond the rank span the complement.
            Eigen::JacobiSVD<Matrix> csvd(defined, Eigen::ComputeFullU);
            complement = csvd.matrixU().rightCols(
                n - static_cast<Eigen::Index>(definedCols.size()));
        }
        for (std::size_t j = 0; j < freeCols.size(); ++j) {
            r.L2.col(freeCols[j]) = complement.col(static_cast<Eigen::Index>(j));
        }
    }

    // R2 follows division-free from the right column block. With the middle
    // factor [[C, -S], [S, C]] we have B = -L1 S R2^H and D = L2 C R2^H, hence
    //   R2^H = -S (L1^H B) + C (L2^H D)     (uses S^2 + C^2 = I).
    const Matrix Sd = sinv.cast<Complex>().asDiagonal();
    const Matrix Cd = cosv.cast<Complex>().asDiagonal();
    const Matrix R2h = -Sd * (r.L1.adjoint() * B) + Cd * (r.L2.adjoint() * D);
    r.R2 = R2h.adjoint();

    return r;
}

Matrix CSDecomposition::reconstruct(const Result& r)
{
    const Eigen::Index n = r.theta.size();

    const Matrix C = r.theta.array().cos().matrix().cast<Complex>().asDiagonal();
    const Matrix S = r.theta.array().sin().matrix().cast<Complex>().asDiagonal();

    Matrix left = Matrix::Zero(2 * n, 2 * n);
    left.topLeftCorner(n, n)     = r.L1;
    left.bottomRightCorner(n, n) = r.L2;

    Matrix mid = Matrix::Zero(2 * n, 2 * n);
    mid.topLeftCorner(n, n)     = C;
    mid.topRightCorner(n, n)    = -S;
    mid.bottomLeftCorner(n, n)  = S;
    mid.bottomRightCorner(n, n) = C;

    Matrix right = Matrix::Zero(2 * n, 2 * n);
    right.topLeftCorner(n, n)     = r.R1;
    right.bottomRightCorner(n, n) = r.R2;

    return left * mid * right.adjoint();
}

} // namespace qsvt
