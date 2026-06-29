#include "KAKDecomposition.hpp"

#include <cmath>
#include <complex>

#include <eigen3/Eigen/Eigenvalues>

namespace qsvt {

using cd = std::complex<double>;

KAKDecomposition::KAKDecomposition(const Eigen::Matrix4cd& matrix)
    : matrix_(matrix)
{
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    const cd i(0.0, 1.0);

    // Magic basis: columns are the (phase-adjusted) Bell states. Conjugating by
    // Q maps SU(2) (x) SU(2) onto the real orthogonal group SO(4).
    Q_ << 1, 0, 0,  i,
          0, i, 1,  0,
          0, i, -1, 0,
          1, 0, 0, -i;
    Q_ *= inv_sqrt_2;

    Q_dagger_ = Q_.adjoint();
}

std::tuple<Eigen::Matrix4cd, Eigen::Matrix4cd, Eigen::Matrix4cd>
KAKDecomposition::solve()
{
    using Eigen::Matrix4cd;
    using Eigen::Matrix4d;
    using Eigen::Vector4cd;

    // 1. Move into the magic basis.
    const Matrix4cd Um = Q_dagger_ * matrix_ * Q_;

    // 2. M = Um^T Um is complex-SYMMETRIC (not Hermitian). Its real and
    //    imaginary parts are real-symmetric and commute, so a single real
    //    orthogonal O diagonalizes both. We diagonalize Mr first, then within
    //    each degenerate eigenspace diagonalize the restriction of Mi. This
    //    handles repeated canonical angles, where a single generic combination
    //    of Mr and Mi would leave M non-diagonal (and the local factors wrong).
    const Matrix4cd M = Um.transpose() * Um;
    const Matrix4d Mr = M.real();
    const Matrix4d Mi = M.imag();

    Eigen::SelfAdjointEigenSolver<Matrix4d> solver(Mr);
    if (solver.info() != Eigen::Success) {
        throw KAKException("eigendecomposition failed");
    }
    const Eigen::Vector4d ev = solver.eigenvalues();   // ascending
    const Matrix4d V = solver.eigenvectors();

    constexpr double kDegenerate = 1e-7;
    Matrix4d O;
    int i = 0;
    while (i < 4) {
        int j = i + 1;
        while (j < 4 && std::abs(ev(j) - ev(i)) < kDegenerate) {
            ++j;
        }
        const int g = j - i; // size of the degenerate block
        if (g == 1) {
            O.col(i) = V.col(i);
        } else {
            const Eigen::MatrixXd block = V.middleCols(i, g);     // 4 x g
            const Eigen::MatrixXd MiSub = block.transpose() * Mi * block;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sub(MiSub);
            if (sub.info() != Eigen::Success) {
                throw KAKException("eigendecomposition failed");
            }
            O.middleCols(i, g) = block * sub.eigenvectors();
        }
        i = j;
    }

    // Force det(O) = +1 so O lives in SO(4) (a reflection would break locality).
    if (O.determinant() < 0.0) {
        O.col(0) *= -1.0;
    }
    const Matrix4cd Oc = O.cast<cd>();

    // 3. Diagonal phases of M in this basis; A's diagonal is their square root.
    const Vector4cd diag = (Oc.transpose() * M * Oc).diagonal();
    Vector4cd a;
    for (int k = 0; k < 4; ++k) {
        a(k) = std::sqrt(diag(k)); // principal branch; |a(k)| = 1
    }

    const Matrix4cd K2m = Oc.transpose();                       // SO(4), det = +1
    Matrix4cd K1m = Um * Oc * a.cwiseInverse().asDiagonal();     // = Um O Am^{-1}

    // K1m is real orthogonal, but only the det = +1 (SO(4)) component maps to a
    // local gate; a det = -1 reflection does not. Flipping one sqrt branch flips
    // det(K1m) while leaving a_k^2 (and hence the reconstruction) unchanged.
    if (K1m.determinant().real() < 0.0) {
        a(0) = -a(0);
        K1m = Um * Oc * a.cwiseInverse().asDiagonal();
    }
    const Matrix4cd Am = a.asDiagonal();

    // 4. Back to the computational basis.
    const Matrix4cd K1 = Q_ * K1m * Q_dagger_;
    const Matrix4cd A  = Q_ * Am  * Q_dagger_;
    const Matrix4cd K2 = Q_ * K2m * Q_dagger_;

    // 5. Interaction angles from the magic-basis phases (ordering-dependent).
    const double t0 = std::arg(a(0));
    const double t1 = std::arg(a(1));
    const double t2 = std::arg(a(2));
    const double t3 = std::arg(a(3));
    angles_.ax = (t0 - t1 - t2 + t3) / 4.0;
    angles_.ay = (t0 - t1 + t2 - t3) / 4.0;
    angles_.az = (t0 + t1 - t2 - t3) / 4.0;

    return std::make_tuple(K1, A, K2);
}

} // namespace qsvt
