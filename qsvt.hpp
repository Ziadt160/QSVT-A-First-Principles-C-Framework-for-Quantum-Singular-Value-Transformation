#pragma once

#include "qrack/qfactory.hpp"

#include <vector>
#include <stdexcept>
#include <functional>


class QSVT 
{
public: 
    /**
     * @brief Constructor for QSVT Class
     * @param num_system_qubits The number of the qubits for the main system
     */
    QSVT(int num_system_qubits);

    /**
     * @brief Destructor. Cleans up the simulator to prevent memory leaks.
     */
    ~QSVT();

    /**
     * @brief Provides direct access to the simulator for state preparation or measurement.
     * @return A pointer to the underlying Qrack::QInterface instance.
    */
    Qrack::QInterfacePtr get_simulator();

private:
    Qrack::QInterfacePtr sim;
    int ancilla_qubit_index;

};