#pragma once
// Quantum Signal Processing (QSP) on a single qubit: interleave a fixed signal
// unitary Ua with Z-rotations parameterized by a sequence of phase angles.

#include <vector>

#include "Common.hpp"
#include "QrackTypes.hpp"

namespace qsvt {

class Qsp {
public:
    /// @param qReg   Simulator interface.
    /// @param Ua     The 2x2 signal unitary, in Qrack layout.
    /// @param angles QSP phase angles (length determines the polynomial degree).
    Qsp(Qrack::QInterfacePtr qReg, QMatrix Ua, std::vector<double> angles);

    /// Apply the QSP sequence to qubit 0.
    void apply();

private:
    std::vector<double> angles_;
    QMatrix Ua_;
    Qrack::QInterfacePtr qReg_;
};

} // namespace qsvt
