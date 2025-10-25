
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include "BlockEncoding/BlockEncoding.hpp"
#include "Qsp/Qsp.hpp"

using namespace Qrack;

int main() {

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL , 1, ZERO_BCI);

    BlockEncoding block_encoding(qReg, 0.3);

    DynamicMatrix Ua = block_encoding.get_encoded_matrix();

    std::vector<double> angles = {
        -7.06858347e+00, 
         7.85398870e-01, 
         7.07098960e-07
    };

    qReg->cc

    Qsp qsp(qReg, Ua, angles);

    qsp.apply();

    std::cout << qReg->Prob(0) << std::endl;

    return 0;
}