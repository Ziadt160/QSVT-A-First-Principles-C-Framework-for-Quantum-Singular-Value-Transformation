#pragma once

#include <Eigen/Eigen>
#include <tuple>

class CSD
{
public:
    /**
     * @brief Constructor for Cosine-Sine Decomposition
     * @param matrix U Matrix (must be of size 2n x 2n)
     */
    explicit CSD(const Eigen::MatrixXcd& matrix);

    /**
     * @brief Computes the CSD.
     * Returns a tuple of (L, D, R) where U = L * D * R.
     * L and R are block diagonal matrices:
     * L = diag(L0, L1)
     * R = diag(R0, R1)
     * D is a matrix formed by blocks of diagonal matrices:
     * D = [ C  -S ]
     *     [ S   C ]
     * where C and S are diagonal matrices with C^2 + S^2 = I.
     */
    std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> solve();

private:
    Eigen::MatrixXcd matrix;
};
