#include "BlockEncoding.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace Qrack;


BlockEncoding::BlockEncoding(QInterfacePtr qReg, double scalar)
{
    if (scalar > 1)
    {
        throw BlockEncodingException("Your scalar value must not exceed 1");
    }

    matrix.resize(2, 2);

    double q = std::sqrt( 1 - std::pow(scalar , 2));

    double p = scalar;

    matrix(0 , 0) = Qrack::complex(p , 0);

    matrix(0 , 1) = Qrack::complex(q, 0);

    matrix(1 , 0) = matrix(0 , 1);
    
    matrix(1, 1) =  Qrack::complex(-1, 0) * matrix(0, 0);

    this->qReg = qReg;
}

BlockEncoding::BlockEncoding(QInterfacePtr qReg, const DynamicMatrix& matrix)
{
    int n = matrix.rows();
    Eigen::JacobiSVD<DynamicMatrix> svd(matrix);

    if ( svd.singularValues()[0] > 1 )
    {
        throw BlockEncodingException(" The matrix norm is larger than 1");
    }

    DynamicMatrix I = DynamicMatrix::Identity(n * 2, n * 2);

}


void BlockEncoding::apply(bitLenInt target) const 
{
    qReg->Mtrx(matrix.data(), target);
}

void BlockEncoding::apply_adjoint(bitLenInt target) const
{
    qReg->Mtrx(matrix.adjoint().eval().data() , target);
}

void BlockEncoding::controlled_apply(const std::vector<uint8_t>& controls, bitLenInt target) const
{
    qReg->MCMtrx(controls, matrix.data(), target);
}

void BlockEncoding::anti_controlled_apply(const std::vector<uint8_t>& controls, bitLenInt target) const
{
    qReg->MACMtrx(controls, matrix.data(), target);
}

const DynamicMatrix& BlockEncoding::get_encoded_matrix() const
{
    return this->matrix;
}

