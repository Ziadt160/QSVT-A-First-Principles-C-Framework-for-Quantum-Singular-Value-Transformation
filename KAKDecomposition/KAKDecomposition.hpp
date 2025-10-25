#pragma once

#include <eigen3/Eigen/Eigen>

using namespace Eigen;

struct AlphaCoefficients {
    double ax, ay, az;
};

class KAKDecomposition
{
public:

    /**
     * @brief Constructor for KAKDecomposition ( Factoring U Matrix into K1AK2)
     * @param matrix U Matrix
     */
    explicit KAKDecomposition(const Eigen::Matrix4cd& matrix);

    /**
     * @brief finding K1, A, K2 from A matrix
     */
    std::tuple<Matrix4cd, Matrix4cd, Matrix4cd> solve();

    /*
    * @brief getter for angles
    */
   AlphaCoefficients getAMatrixAngles() const;

private:
    Eigen::Matrix4cd matrix;
    Eigen::Matrix4cd Q;
    Eigen::Matrix4cd Q_dagger;
    AlphaCoefficients angles;
};