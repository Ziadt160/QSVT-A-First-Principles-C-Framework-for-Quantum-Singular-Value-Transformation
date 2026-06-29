#include "Qsvt.hpp"

namespace qsvt {

QSVT::QSVT(int num_system_qubits)
{
    if (num_system_qubits <= 0) {
        throw std::invalid_argument("Number of system qubits must be positive.");
    }

    // Total qubits = system qubits + 1 ancilla for the block-encoding.
    const int total_qubits = num_system_qubits + 1;
    sim_ = Qrack::CreateQuantumInterface(Qrack::QINTERFACE_OPTIMAL,
                                         total_qubits, Qrack::ZERO_BCI);

    // The last qubit is the ancilla.
    ancilla_qubit_index_ = num_system_qubits;
}

} // namespace qsvt
