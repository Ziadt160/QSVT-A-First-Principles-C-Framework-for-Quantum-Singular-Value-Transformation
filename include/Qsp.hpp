#pragma once

#include <qrack/qfactory.hpp>
#include <eigen3/Eigen/Eigen>

using namespace Qrack;

typedef Eigen::Matrix<Qrack::complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DynamicMatrix;


class Qsp
{
private:
    std::vector<double> angles;
    DynamicMatrix Ua;
    QInterfacePtr qReg;

public:

    explicit Qsp(QInterfacePtr qReg, DynamicMatrix Ua, std::vector<double> angles);
    void apply();
};