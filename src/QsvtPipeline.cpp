#include "QsvtPipeline.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

#include <eigen3/Eigen/Eigenvalues>

#include "SymQspAngleSolver.hpp"
#include "ShannonDecomposition.hpp"

namespace qsvt {

namespace {
const Complex kI(0.0, 1.0);
}

Matrix rotationBlockEncoding(const Matrix& A)
{
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("rotationBlockEncoding: A must be square");
    }
    if ((A - A.adjoint()).norm() > 1e-9) {
        throw std::invalid_argument("rotationBlockEncoding: A must be Hermitian");
    }

    const Eigen::Index k = A.rows();
    Eigen::SelfAdjointEigenSolver<Matrix> es(A);
    const RealVector lambda = es.eigenvalues();
    if (lambda.cwiseAbs().maxCoeff() > 1.0 + kTolerance) {
        throw std::invalid_argument("rotationBlockEncoding: ||A|| must be <= 1");
    }

    // B = sqrt(I - A^2), built in A's eigenbasis.
    RealVector s = (1.0 - lambda.array().square()).cwiseMax(0.0).sqrt();
    const Matrix V = es.eigenvectors();
    const Matrix B = V * s.cast<Complex>().asDiagonal() * V.adjoint();

    Matrix Uw(2 * k, 2 * k);
    Uw.topLeftCorner(k, k)     = A;
    Uw.topRightCorner(k, k)    = kI * B;
    Uw.bottomLeftCorner(k, k)  = kI * B;
    Uw.bottomRightCorner(k, k) = A;
    return Uw;
}

Matrix qsvtUnitary(const Matrix& A, const std::vector<double>& phases)
{
    const Matrix Uw = rotationBlockEncoding(A);
    const Eigen::Index k = A.rows();
    const Eigen::Index n = 2 * k;

    // E(phi) = e^{i phi Z_anc} = diag(e^{i phi} I_k, e^{-i phi} I_k).
    auto E = [&](double phi) {
        Matrix m = Matrix::Zero(n, n);
        for (Eigen::Index j = 0; j < k; ++j) m(j, j) = std::exp(kI * phi);
        for (Eigen::Index j = k; j < n; ++j) m(j, j) = std::exp(-kI * phi);
        return m;
    };

    Matrix U = E(phases[0]);
    for (std::size_t i = 1; i < phases.size(); ++i) {
        U = U * Uw * E(phases[i]);
    }
    return U;
}

Matrix qsvtBlock(const Matrix& A, const std::vector<double>& phases)
{
    const Eigen::Index k = A.rows();
    return qsvtUnitary(A, phases).topLeftCorner(k, k);
}

std::vector<Gate> qsvtCircuit(const Matrix& A, const std::vector<double>& phases)
{
    const Matrix Uw = rotationBlockEncoding(A);
    const int nQubits = static_cast<int>(
        std::lround(std::log2(static_cast<double>(Uw.rows()))));
    const int ancilla = nQubits - 1; // block-encoding ancilla = highest qubit

    // E(phi) = e^{i phi Z} on the ancilla = diag(e^{i phi}, e^{-i phi}).
    auto phaseGate = [&](double phi) {
        Eigen::Matrix2cd m = Eigen::Matrix2cd::Zero();
        m(0, 0) = std::exp(kI * phi);
        m(1, 1) = std::exp(-kI * phi);
        return Gate::single(m, ancilla);
    };

    // Compile U_W once and reuse.
    const std::vector<Gate> uw = ShannonDecomposition(Uw).gates();

    std::vector<Gate> circuit;
    circuit.push_back(phaseGate(phases[0]));
    for (std::size_t i = 1; i < phases.size(); ++i) {
        circuit.insert(circuit.end(), uw.begin(), uw.end());
        circuit.push_back(phaseGate(phases[i]));
    }
    return circuit;
}

QsvtProgram compileMatrixFunction(const Matrix& A,
                                  const std::function<double(double)>& target,
                                  int degree)
{
    QsvtProgram p;
    const QspSolveResult sol = SymQspAngleSolver(degree).solve(target);
    p.phases = sol.phases;
    p.polyResidual = sol.residual;
    p.converged = sol.converged;
    p.circuit = qsvtCircuit(A, p.phases);
    p.numQubits = static_cast<int>(
        std::lround(std::log2(static_cast<double>(2 * A.rows()))));
    p.resources = countResources(p.circuit, p.numQubits);
    return p;
}

} // namespace qsvt
