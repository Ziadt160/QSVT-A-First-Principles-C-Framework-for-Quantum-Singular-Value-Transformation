#pragma once
// A native gate and the helpers to evaluate / run a gate list. Gates reference
// "logical" qubit indices (0 = least significant); a circuit is mapped onto
// physical Qrack qubits at run time. Shared by the 2-qubit and n-qubit
// synthesizers.

#include <cstddef>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "Common.hpp"
#include "QrackTypes.hpp"

namespace qsvt {

/// Either a single-qubit 2x2 unitary on `target`, or a CNOT from `control` to
/// `target`.
struct Gate {
    bool isCnot{false};
    Eigen::Matrix2cd matrix{Eigen::Matrix2cd::Identity()};
    int control{0};
    int target{0};

    static Gate single(const Eigen::Matrix2cd& m, int target)
    {
        Gate g;
        g.isCnot = false;
        g.matrix = m;
        g.target = target;
        return g;
    }
    static Gate cnot(int control, int target)
    {
        Gate g;
        g.isCnot = true;
        g.control = control;
        g.target = target;
        return g;
    }
};

/// Dense 2^n x 2^n unitary implemented by the gate list (logical qubit 0 = least
/// significant). Gates are applied in list order.
Matrix denseCircuit(const std::vector<Gate>& gates, int nQubits);

/// Apply the gate list to a Qrack register; logical qubit i maps to qubits[i].
void runCircuit(const std::vector<Gate>& gates, Qrack::QInterfacePtr qReg,
                const std::vector<bitLenInt>& qubits);

/// Peephole optimisation that preserves the implemented unitary exactly:
/// cancels adjacent identical CNOTs (including across block boundaries) and
/// fuses adjacent single-qubit gates on the same qubit into one.
std::vector<Gate> optimizeCircuit(const std::vector<Gate>& gates);

/// Gate-count and depth summary of a circuit.
struct ResourceCounts {
    std::size_t cnot{0};
    std::size_t singleQubit{0};
    std::size_t total{0};
    std::size_t depth{0}; // critical-path length (ASAP scheduling)
};

ResourceCounts countResources(const std::vector<Gate>& gates, int nQubits);

/// Emit the circuit as an OpenQASM 2.0 program. Single-qubit gates become
/// `U(theta,phi,lambda)` (via a ZYZ Euler factorisation) and CNOTs become `cx`.
std::string toQasm(const std::vector<Gate>& gates, int nQubits);

} // namespace qsvt
