#include "EigenvalueThreshold.hpp"

#include <cmath>

#include "QspAngleSolver.hpp"
#include "QsvtPipeline.hpp"

namespace qsvt {

ThresholdProgram compileEigenvalueThreshold(const Matrix& H, double mu, double w,
                                            int degree)
{
    ThresholdProgram p;
    p.mu = mu;

    // sign is odd: use the largest odd degree <= degree.
    const int d = (degree % 2 == 1) ? degree : degree - 1;
    const Matrix A = H - mu * Matrix::Identity(H.rows(), H.cols());

    const QspSolveResult s =
        QspAngleSolver(d).solve([w](double x) { return std::erf(x / w); });
    p.phases = s.phases;
    p.fitResidual = s.residual;
    p.converged = s.converged;

    p.circuit = qsvtCircuit(A, p.phases);
    p.numQubits = static_cast<int>(
        std::lround(std::log2(static_cast<double>(2 * A.rows()))));
    p.resources = countResources(p.circuit, p.numQubits);
    return p;
}

Matrix spectralProjector(const Matrix& H, const ThresholdProgram& program)
{
    const Matrix A = H - program.mu * Matrix::Identity(H.rows(), H.cols());
    const Matrix block = qsvtBlock(A, program.phases); // P_sign(A)
    const Matrix signA = 0.5 * (block + block.adjoint()); // = sign(H - mu)
    return 0.5 * (Matrix::Identity(H.rows(), H.cols()) + signA);
}

} // namespace qsvt
