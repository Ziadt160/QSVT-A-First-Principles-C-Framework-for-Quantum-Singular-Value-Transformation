#pragma once
// Top-level QSVT engine: owns the simulator and the ancilla bookkeeping that the
// block-encoding / QSP pipeline is built on.

#include <stdexcept>

#include <qrack/qfactory.hpp>

namespace qsvt {

class QSVT {
public:
    /// @param num_system_qubits Number of qubits for the main system (an extra
    ///        ancilla qubit is allocated on top for the block-encoding).
    explicit QSVT(int num_system_qubits);

    /// Direct access to the simulator for state preparation or measurement.
    Qrack::QInterfacePtr get_simulator() const { return sim_; }

    /// Index of the ancilla qubit used by the block-encoding.
    int ancilla_index() const { return ancilla_qubit_index_; }

private:
    Qrack::QInterfacePtr sim_;
    int ancilla_qubit_index_;
};

} // namespace qsvt
