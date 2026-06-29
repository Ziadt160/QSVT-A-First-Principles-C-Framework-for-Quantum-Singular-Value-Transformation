#pragma once
// Hamiltonian simulation via QSVT: approximate the time-evolution operator
// e^{-iHt} for a Hermitian H with ||H|| <= 1.
//
//   e^{-iHt} = cos(tH) - i sin(tH).
//
// cos(t x) (even) and sin(t x) (odd) are bounded by 1 on [-1, 1] and have the
// rapidly-converging Jacobi-Anger / Chebyshev expansions, so QspAngleSolver
// fits each, and QSVT applies the resulting polynomial to the eigenvalues of H.
// Each QSVT block U_Phi gives P(H) with Re P(lambda) = the target; the Hermitian
// part (block + block^H)/2 then recovers cos(tH) or sin(tH) exactly. Combining
// the two halves (an LCU on hardware) yields the evolution operator.

#include <functional>
#include <vector>

#include "Common.hpp"
#include "Gate.hpp"

namespace qsvt {

struct HamSimProgram {
    std::vector<double> cosPhases, sinPhases;
    std::vector<Gate> cosCircuit, sinCircuit; // QSVT circuits for cos(tH), sin(tH)
    int numQubits{0};                         // system + 1 ancilla
    ResourceCounts cosResources, sinResources;
    double cosFitResidual{0.0}, sinFitResidual{0.0};
    bool converged{false};
};

/// Compile the two QSVT circuits realising cos(tH) and sin(tH). `degree` sets the
/// polynomial truncation (~ t + a few suffices for good accuracy).
HamSimProgram compileHamiltonianSimulation(const Matrix& H, double t, int degree);

/// Dense approximation of e^{-iHt} built from the program's QSVT blocks:
/// HermPart(block_cos) - i * HermPart(block_sin).
Matrix simulatedEvolution(const Matrix& H, const HamSimProgram& program);

} // namespace qsvt
