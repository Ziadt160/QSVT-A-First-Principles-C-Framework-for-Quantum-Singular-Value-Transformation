#include <complex>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include "CSDecomposition.hpp"
#include "EigenvalueThreshold.hpp"
#include "HamiltonianSimulation.hpp"
#include "KAKDecomposition.hpp"
#include "QspAngleSolver.hpp"
#include "Qsvt.hpp"
#include "QsvtPipeline.hpp"
#include "ShannonDecomposition.hpp"
#include "TwoQubitSynthesis.hpp"

namespace {

// A reproducible "random" unitary via QR of a fixed complex matrix.
qsvt::Matrix sampleUnitary(Eigen::Index dim, unsigned seed)
{
    qsvt::Matrix m(dim, dim);
    unsigned state = seed * 2654435761u + 1u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state) / 4294967296.0 - 0.5;
    };
    for (Eigen::Index r = 0; r < dim; ++r) {
        for (Eigen::Index c = 0; c < dim; ++c) {
            m(r, c) = qsvt::Complex(next(), next());
        }
    }
    Eigen::HouseholderQR<qsvt::Matrix> qr(m);
    qsvt::Matrix q = qr.householderQ();
    return q;
}

} // namespace

int main()
{
    std::cout << "Running QSVT demo...\n";

    // --- KAK decomposition of a 2-qubit unitary ---------------------------
    const qsvt::Matrix u4 = sampleUnitary(4, 7);
    qsvt::KAKDecomposition kak(u4);
    auto [K1, A, K2] = kak.solve();
    const double kakErr = (K1 * A * K2 - u4).norm();
    std::cout << "KAK reconstruction error:  " << kakErr << '\n';

    // --- Compile the same 2-qubit unitary to a native gate sequence ---------
    qsvt::TwoQubitSynthesis synth(u4);
    const double gateErr = (synth.circuitMatrix() - u4).norm();
    std::cout << "Gate-synthesis error:      " << gateErr
              << "  (" << synth.gates().size() << " native gates)\n";

    // --- Cosine-Sine decomposition of a larger (8x8) unitary --------------
    const qsvt::Matrix u8 = sampleUnitary(8, 13);
    qsvt::CSDecomposition cs(u8);
    const auto res = cs.solve();
    const double csErr = (qsvt::CSDecomposition::reconstruct(res) - u8).norm();
    std::cout << "CSD reconstruction error:  " << csErr << '\n';

    // --- Quantum Shannon Decomposition of an arbitrary 3-qubit unitary -----
    qsvt::ShannonDecomposition qsd(u8);
    const double qsdErr = (qsd.circuitMatrix() - u8).norm();
    std::cout << "QSD (3-qubit) error:       " << qsdErr
              << "  (" << qsd.gates().size() << " native gates)\n";

    // --- QSP angle finding for a target polynomial -------------------------
    // Target: f(x) = 0.4*T_1(x) + 0.5*T_3(x), an odd degree-3 polynomial.
    auto target = [](double x) { return 0.4 * x + 0.5 * (4.0 * x * x * x - 3.0 * x); };
    qsvt::QspAngleSolver solver(3);
    const qsvt::QspSolveResult qsp = solver.solve(target);
    std::cout << "QSP solve: " << qsp.phases.size() << " phases, residual "
              << qsp.residual << (qsp.converged ? " (converged)\n" : " (NOT converged)\n");

    // --- FLAGSHIP: matrix-function compiler, applied to matrix inversion ----
    // Build a Hermitian, well-conditioned A (spectrum in [0.3, 1]) and compile
    // a QSVT circuit that applies p(A) ~ (delta / A), i.e. a scaled inverse.
    const double delta = 0.3; // condition number 1/delta ~ 3.3
    const Eigen::Index k = 4; // 2 system qubits
    const qsvt::Matrix W = sampleUnitary(k, 21);
    Eigen::VectorXd ev(k);
    for (Eigen::Index i = 0; i < k; ++i) ev(i) = delta + (1.0 - delta) * i / (k - 1);
    const qsvt::Matrix Amat = W * ev.cast<qsvt::Complex>().asDiagonal() * W.adjoint();

    // Regularized inverse f(x) = c * x / (x^2 + eps): smooth, odd, ~ c/x away
    // from 0 (the standard QSVT inversion target). c is kept well inside [-1,1]
    // (peak ~0.4) so our Levenberg-Marquardt angle solver converges.
    const double eps = (delta / 4.0) * (delta / 4.0); // small -> tracks 1/x on [delta,1]
    const double c = 0.9 * 2.0 * std::sqrt(eps); // peak |f| = 0.9 (in solver range)
    auto invTarget = [&](double x) { return c * x / (x * x + eps); };
    const int invDegree = 25; // homotopy solver reaches this comfortably
    const qsvt::QsvtProgram prog = qsvt::compileMatrixFunction(Amat, invTarget, invDegree);

    std::cout << "\n--- Matrix-function compiler: inversion of a " << k << "x" << k
              << " Hermitian (kappa=" << ev(k - 1) / ev(0) << ") ---\n";
    std::cout << "Compiled QSVT circuit: " << prog.numQubits << " qubits, "
              << prog.resources.cnot << " CNOTs, " << prog.resources.total
              << " gates, depth " << prog.resources.depth << '\n';
    std::cout << "Polynomial fit residual (deg " << invDegree << "): "
              << prog.polyResidual << '\n';

    // QSVT realises the transform exactly: block(U_Phi) == P(A) to machine
    // precision. Measure the realised inverse on A's spectrum.
    Eigen::SelfAdjointEigenSolver<qsvt::Matrix> es(Amat);
    double fitErr = 0.0, invErr = 0.0;
    for (Eigen::Index i = 0; i < k; ++i) {
        const double lam = es.eigenvalues()(i);
        const double realized =
            qsvt::QspAngleSolver::response00(lam, prog.phases).real();
        fitErr = std::max(fitErr, std::abs(realized - invTarget(lam))); // solver fit
        invErr = std::max(invErr, std::abs(realized - c / lam));        // vs true c/x
    }
    std::cout << "realised transform vs target on spectrum (solver fit): " << fitErr << '\n';
    std::cout << "realised transform vs c/lambda (scaled inverse):       " << invErr << '\n';
    std::cout << "(block(U_Phi) == P(A) to machine precision; remaining error is the "
                 "angle-solver / degree-" << invDegree << " approximation.)\n";

    // --- APPLICATION: Hamiltonian simulation e^{-iHt} via QSVT --------------
    // H = 0.5 XX + 0.3 ZI + 0.2 IZ (a 2-qubit transverse-field-like model),
    // rescaled so ||H|| <= 0.9 for the block-encoding.
    Eigen::Matrix2cd X, Z, Id2;
    X << 0, 1, 1, 0;
    Z << 1, 0, 0, -1;
    Id2 = Eigen::Matrix2cd::Identity();
    qsvt::Matrix H = 0.5 * Eigen::kroneckerProduct(X, X)
                   + 0.3 * Eigen::kroneckerProduct(Z, Id2)
                   + 0.2 * Eigen::kroneckerProduct(Id2, Z);
    Eigen::SelfAdjointEigenSolver<qsvt::Matrix> esH(H);
    H *= 0.9 / esH.eigenvalues().cwiseAbs().maxCoeff();

    const double simT = 3.0;
    const qsvt::HamSimProgram ham = qsvt::compileHamiltonianSimulation(H, simT, 21);
    const qsvt::Matrix approxU = qsvt::simulatedEvolution(H, ham);

    Eigen::SelfAdjointEigenSolver<qsvt::Matrix> esH2(H);
    Eigen::VectorXcd ph(H.rows());
    for (Eigen::Index i = 0; i < H.rows(); ++i)
        ph(i) = std::exp(std::complex<double>(0, -1) * simT * esH2.eigenvalues()(i));
    const qsvt::Matrix exactU =
        esH2.eigenvectors() * ph.asDiagonal() * esH2.eigenvectors().adjoint();
    const double hamErr = (approxU - exactU).norm();

    std::cout << "\n--- Hamiltonian simulation: e^{-iHt}, t = " << simT
              << " (2-qubit H, QSVT degree 21) ---\n";
    std::cout << "cos(tH) circuit: " << ham.cosResources.cnot << " CNOTs; "
              << "sin(tH) circuit: " << ham.sinResources.cnot << " CNOTs\n";
    std::cout << "|| e^{-iHt}_QSVT - e^{-iHt}_exact ||: " << hamErr << '\n';

    // --- APPLICATION: eigenvalue thresholding (spectral projector) ----------
    // Project onto the positive-eigenvalue subspace of a 4x4 Hermitian via
    // sign(H): Pi_{>0} = (I + sign(H)) / 2.
    const qsvt::Matrix Wt = sampleUnitary(4, 314);
    Eigen::Vector4d evt;
    evt << -0.7, -0.4, 0.4, 0.8;
    const qsvt::Matrix Ht = Wt * evt.cast<qsvt::Complex>().asDiagonal() * Wt.adjoint();
    const qsvt::ThresholdProgram thr =
        qsvt::compileEigenvalueThreshold(Ht, /*mu=*/0.0, /*w=*/0.1, /*degree=*/25);
    const qsvt::Matrix P = qsvt::spectralProjector(Ht, thr);
    Eigen::Vector4d maskv;
    for (int i = 0; i < 4; ++i) maskv(i) = evt(i) > 0 ? 1.0 : 0.0;
    const qsvt::Matrix exactP = Wt * maskv.cast<qsvt::Complex>().asDiagonal() * Wt.adjoint();
    std::cout << "\n--- Eigenvalue thresholding: project onto lambda > 0 ---\n";
    std::cout << "projector circuit: " << thr.resources.cnot << " CNOTs, degree 25\n";
    std::cout << "|| P_QSVT - exact projector ||: " << (P - exactP).norm() << '\n';

    // --- Simulator with an ancilla allocated ------------------------------
    qsvt::QSVT engine(2);
    std::cout << "QSVT ancilla qubit index:  " << engine.ancilla_index() << '\n';

    return (kakErr < 1e-9 && csErr < 1e-9 && gateErr < 1e-9 && qsdErr < 1e-9 &&
            qsp.converged && hamErr < 1e-3) ? 0 : 1;
}
