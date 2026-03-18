#include "KAKDecomposition.hpp"
#include <eigen3/Eigen/Eigen>


KAKDecomposition::KAKDecomposition(const Matrix4cd& matrix)
{
    this->matrix = matrix;
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    const std::complex<double> i(0.0, 1.0);

    this->Q << 1, 0, 0,  i,
         0, i, 1,  0,
         0, i, -1, 0,
         1, 0, 0, -i;

    this->Q *= inv_sqrt_2;

    this->Q_dagger = this->Q.adjoint();
}

std::tuple<Matrix4cd, Matrix4cd, Matrix4cd> KAKDecomposition::solve()
{
    Matrix4cd U_magic;

    U_magic = (Q.adjoint().eval() * matrix * Q);

    Matrix4cd M;

    M = U_magic.transpose().eval() * U_magic;

    Matrix4d M_R = M.real();
    Matrix4d M_I = M.imag();
    Matrix4d H_real = M_R + 1.23456789 * M_I;
    SelfAdjointEigenSolver<Matrix4d> eigen_solver(H_real);

    if(eigen_solver.info() != Success)
    {
        std::runtime_error("Eigendecomposition failed!");
    }

    Matrix4d O2_real = eigen_solver.eigenvectors();

    if (O2_real.determinant() < 0.0) {
        O2_real.col(0) *= -1.0;
    }

    Matrix4cd D = O2_real.transpose().cast<std::complex<double>>() * M * O2_real.cast<std::complex<double>>();

    Vector4cd D_diag = D.diagonal();
    Vector4cd sqrt_D = D_diag.cwiseSqrt();
    Matrix4cd A_m = sqrt_D.asDiagonal();

    Matrix4cd O1_complex = U_magic * O2_real.cast<std::complex<double>>() * A_m.inverse();
    Matrix4d O1_real = O1_complex.real();

    if (O1_real.determinant() < 0.0) {
        O1_real.col(0) *= -1.0;
        A_m(0, 0) *= -1.0;
    }

    Matrix4cd K1 = Q * O1_real.cast<std::complex<double>>() * Q_dagger;
    
    Matrix4cd A  = Q * A_m * Q_dagger;
    
    Matrix4cd K2 = Q * O2_real.transpose().cast<std::complex<double>>() * Q_dagger;

    // To get the angles, we can inspect A_m which contains the diagonal eigenvalues of the magic basis.
    // The entries are exp(i * theta_j). We extract angles.
    Vector4cd am_diag = A_m.diagonal();
    double t1 = std::arg(am_diag(0));
    double t2 = std::arg(am_diag(1));
    double t3 = std::arg(am_diag(2));
    double t4 = std::arg(am_diag(3));

    angles.ax = (t1 - t2 - t3 + t4) / 4.0;
    angles.ay = (t1 - t2 + t3 - t4) / 4.0;
    angles.az = (t1 + t2 - t3 - t4) / 4.0;

    return std::make_tuple(K1, A, K2);
}

AlphaCoefficients KAKDecomposition::getAMatrixAngles() const
{
    return angles;
}