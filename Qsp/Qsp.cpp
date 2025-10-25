#include "Qsp.hpp"

Qsp::Qsp(QInterfacePtr qReg, DynamicMatrix Ua, std::vector<double> angles)
{
    this->qReg = qReg;
    this->Ua = Ua;
    this->angles = angles;
}

void Qsp::apply()
{
    qReg->RZ( -2 * angles[0], 0);

    for( int k = 1; k < angles.size(); k = k + 1)
    {
        qReg->Mtrx(Ua.data(), 0);

        qReg->RZ(-2 * angles[k] , 0);
    }
    
}