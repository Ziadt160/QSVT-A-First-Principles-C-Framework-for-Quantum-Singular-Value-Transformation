#include "HamiltonianSimulation.hpp"

#include <cmath>
#include <complex>

#include "QsvtPipeline.hpp"
#include "SymQspAngleSolver.hpp"

namespace qsvt {

HamSimProgram compileHamiltonianSimulation(const Matrix& H, double t, int degree)
{
    HamSimProgram p;

    // cos is even, sin is odd: pick the largest degree of each parity <= degree.
    const int cosDeg = (degree % 2 == 0) ? degree : degree - 1;
    const int sinDeg = (degree % 2 == 1) ? degree : degree - 1;

    const QspSolveResult cs =
        SymQspAngleSolver(cosDeg).solve([t](double x) { return std::cos(t * x); });
    const QspSolveResult ss =
        SymQspAngleSolver(sinDeg).solve([t](double x) { return std::sin(t * x); });

    p.cosPhases = cs.phases;
    p.sinPhases = ss.phases;
    p.cosFitResidual = cs.residual;
    p.sinFitResidual = ss.residual;
    p.converged = cs.converged && ss.converged;

    p.cosCircuit = qsvtCircuit(H, p.cosPhases);
    p.sinCircuit = qsvtCircuit(H, p.sinPhases);
    p.numQubits = static_cast<int>(
        std::lround(std::log2(static_cast<double>(2 * H.rows()))));
    p.cosResources = countResources(p.cosCircuit, p.numQubits);
    p.sinResources = countResources(p.sinCircuit, p.numQubits);
    return p;
}

Matrix simulatedEvolution(const Matrix& H, const HamSimProgram& program)
{
    const Complex I(0.0, 1.0);
    const Matrix bc = qsvtBlock(H, program.cosPhases); // P_cos(H)
    const Matrix bs = qsvtBlock(H, program.sinPhases); // P_sin(H)

    // Hermitian part of each block recovers the real matrix function exactly.
    const Matrix cosH = 0.5 * (bc + bc.adjoint()); // = cos(tH)
    const Matrix sinH = 0.5 * (bs + bs.adjoint()); // = sin(tH)
    return cosH - I * sinH;                         // ~ e^{-iHt}
}

} // namespace qsvt
