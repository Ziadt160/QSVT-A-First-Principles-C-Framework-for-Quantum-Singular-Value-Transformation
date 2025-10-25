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

    SelfAdjointEigenSolver<Matrix4cd> eigen_solver(M);

    if(eigen_solver.info() != Success)
    {
        std::runtime_error("Eigendecomposition failed!");
    }

    Vector4cd eigenvalues = eigen_solver.eigenvalues();

    // Vector4cd sqrt_eigenvalues = eigenvalues.cwiseSqrt();

    Matrix4cd Am_real = eigenvalues.asDiagonal();

    Matrix4cd P = eigen_solver.eigenvectors();

    if (Am_real.determinant().real() < 0.00) {
        Am_real(0, 0) *= -1.0;
    }

    Matrix4cd O2_real = P;

    Matrix4cd O1_real = U_magic * O2_real.transpose().eval() * Am_real.cwiseInverse();

    Matrix4cd O1_complex = O1_real.cast<std::complex<double>>();

    Matrix4cd Am_complex = Am_real.cast<std::complex<double>>();

    Matrix4cd O2_complex = O2_real.cast<std::complex<double>>();

    Matrix4cd K1 = Q * O1_complex * Q_dagger;
    
    Matrix4cd A  = Q * Am_complex * Q_dagger;
    
    Matrix4cd K2 = Q * O2_complex * Q_dagger;

    return std::make_tuple(K1, A , K2);

    // std::vector<double> thetas;

    // for(auto const &singular_value : singular_values)
    // {
    //     thetas.push_back( std::arg(singular_value));
    // }


    // double t1 = thetas[0];
    // double t2 = thetas[1];
    // double t3 = thetas[2];
    // double t4 = thetas[3];

    // angles.ax = (t1 - t2 - t3 + t4) / 4.0;
    // angles.ay = (t1 - t2 + t3 - t4) / 4.0;
    // angles.az = (t1 + t2 - t3 - t4) / 4.0;
}

AlphaCoefficients KAKDecomposition::getAMatrixAngles() const
{
    return angles;
}