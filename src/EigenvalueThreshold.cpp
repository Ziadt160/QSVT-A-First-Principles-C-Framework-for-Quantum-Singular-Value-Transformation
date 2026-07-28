#include "EigenvalueThreshold.hpp"

#include <cmath>

#include "QsvtPipeline.hpp"
#include "InverseApproximation.hpp"
#include "SymQspAngleSolver.hpp"

namespace qsvt {

ThresholdProgram compileEigenvalueThreshold(const Matrix& H, double mu, double w,
                                            int degree)
{
    ThresholdProgram p;
    p.mu = mu;

    // sign is odd: use the largest odd degree <= degree.
    const int d = (degree % 2 == 1) ? degree : degree - 1;
    const Matrix A = H - mu * Matrix::Identity(H.rows(), H.cols());

    // erf tends to +-1, so its degree-d Chebyshev truncation overshoots |p| = 1
    // (e.g. sup = 1.021 at w = 0.1, d = 25). A symmetric-QSP solve on such a
    // target has no solution, so shave it to a valid one and divide the scale
    // back out when forming the projector.
    auto raw = [w](double x) { return std::erf(x / w); };
    const double scale = chebyshevSafetyScale(raw, d);
    const QspSolveResult s =
        SymQspAngleSolver(d).solve([&raw, scale](double x) { return scale * raw(x); });
    p.phases = s.phases;
    p.fitResidual = s.residual;
    p.converged = s.converged;
    p.targetScale = scale;

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
    // QSVT realises targetScale * sign(H - mu); divide the subnormalization out.
    const Matrix signA =
        0.5 * (block + block.adjoint()) / program.targetScale; // = sign(H - mu)
    return 0.5 * (Matrix::Identity(H.rows(), H.cols()) + signA);
}

} // namespace qsvt
