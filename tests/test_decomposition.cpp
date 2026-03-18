#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>
#include "KAKDecomposition.hpp"
#include "CSD.hpp"

using namespace Eigen;

TEST(DecompositionTest, KAKReconstruction) {
    Matrix4cd X = Matrix4cd::Random();
    Matrix4cd H = X + X.adjoint();
    Matrix4cd U = (std::complex<double>(0, 1) * H).exp();

    KAKDecomposition kak(U);
    auto [K1, A, K2] = kak.solve();

    Matrix4cd U_recon = K1 * A * K2;

    double diff = (U - U_recon).norm();
    EXPECT_LT(diff, 1e-10);
}

TEST(DecompositionTest, CSDReconstruction4x4) {
    Matrix4cd X = Matrix4cd::Random();
    Matrix4cd H = X + X.adjoint();
    Matrix4cd U = (std::complex<double>(0, 1) * H).exp();

    CSD csd(U);
    auto [L, D, R] = csd.solve();

    Matrix4cd U_recon = L * D * R;

    double diff = (U - U_recon).norm();
    EXPECT_LT(diff, 1e-10);
}

TEST(DecompositionTest, CSDReconstruction8x8) {
    int N = 8;
    MatrixXcd X = MatrixXcd::Random(N, N);
    MatrixXcd H = X + X.adjoint();
    MatrixXcd U = (std::complex<double>(0, 1) * H).exp();

    CSD csd(U);
    auto [L, D, R] = csd.solve();

    MatrixXcd U_recon = L * D * R;

    double diff = (U - U_recon).norm();
    EXPECT_LT(diff, 1e-10);
}
