#include "qsvt.hpp"

using namespace Qrack;

QSVT::QSVT(int num_system_qubits) {
    if (num_system_qubits <= 0) {
        throw std::invalid_argument("Number of system qubits must be positive.");
    }
    // Total qubits = system qubits + 1 for the ancilla.
    sim = CreateQuantumInterface(QINTERFACE_OPTIMAL, num_system_qubits, ZERO_BCI);
    // We'll use the last qubit as our ancilla for convenience.
    ancilla_qubit_index = num_system_qubits;
}

QSVT::~QSVT() {
}

Qrack::QInterfacePtr QSVT::get_simulator() {
    return sim;
}