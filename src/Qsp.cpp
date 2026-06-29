#include "Qsp.hpp"

#include <utility>

namespace qsvt {

Qsp::Qsp(Qrack::QInterfacePtr qReg, QMatrix Ua, std::vector<double> angles)
    : angles_(std::move(angles)), Ua_(std::move(Ua)), qReg_(std::move(qReg))
{
}

void Qsp::apply()
{
    if (angles_.empty()) {
        return;
    }

    qReg_->RZ(-2.0 * angles_[0], 0);

    for (std::size_t k = 1; k < angles_.size(); ++k) {
        qReg_->Mtrx(Ua_.data(), 0);
        qReg_->RZ(-2.0 * angles_[k], 0);
    }
}

} // namespace qsvt
