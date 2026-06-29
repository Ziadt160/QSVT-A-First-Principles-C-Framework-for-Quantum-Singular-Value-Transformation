#include <complex>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <qrack/qfactory.hpp>

#include "BlockEncoding.hpp"
#include "CSDecomposition.hpp"
#include "KAKDecomposition.hpp"
#include "EigenvalueThreshold.hpp"
#include "HamiltonianSimulation.hpp"
#include "Lcu.hpp"
#include "Qsp.hpp"
#include "QspAngleSolver.hpp"
#include "Qsvt.hpp"
#include "QsvtPipeline.hpp"
#include "ShannonDecomposition.hpp"
#include "TwoQubitSynthesis.hpp"

using namespace Qrack;

namespace {

// Deterministic "random" unitary via QR of a seeded complex matrix.
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
    return qsvt::Matrix(qr.householderQ());
}

bool isUnitary(const qsvt::Matrix& m, double tol = 1e-9)
{
    const qsvt::Matrix id = qsvt::Matrix::Identity(m.rows(), m.cols());
    return (m.adjoint() * m - id).norm() < tol;
}

// Hermitian matrix U diag(eigs) U^H with eigenvalues drawn in [-maxAbs, maxAbs].
qsvt::Matrix sampleHermitian(Eigen::Index dim, unsigned seed, double maxAbs)
{
    const qsvt::Matrix U = sampleUnitary(dim, seed);
    Eigen::VectorXd eigs(dim);
    unsigned state = seed * 40503u + 7u;
    for (Eigen::Index i = 0; i < dim; ++i) {
        state = state * 1664525u + 1013904223u;
        eigs(i) = maxAbs * (2.0 * (static_cast<double>(state) / 4294967296.0) - 1.0);
    }
    return U * eigs.cast<qsvt::Complex>().asDiagonal() * U.adjoint();
}

// Chebyshev polynomial of the first kind, T_k(x).
double chebT(int k, double x)
{
    if (k == 0) return 1.0;
    if (k == 1) return x;
    double a = 1.0, b = x;
    for (int i = 2; i <= k; ++i) {
        const double c = 2.0 * x * b - a;
        a = b;
        b = c;
    }
    return b;
}

// Compare two state vectors that may differ by an unobservable global phase.
bool equalUpToGlobalPhase(const Eigen::VectorXcd& v, const Eigen::VectorXcd& u,
                          double tol)
{
    Eigen::Index r = 0;
    u.cwiseAbs().maxCoeff(&r);
    std::complex<double> ph = v(r) / u(r);
    ph /= std::abs(ph);
    return (v - ph * u).norm() < tol;
}

// Run a circuit-bearing object's applyTo on a Qrack register for a random input
// state and return the resulting vector (in double precision).
template <typename ApplyFn>
Eigen::VectorXcd simulate(int nQubits, const Eigen::VectorXcd& psi, ApplyFn apply)
{
    const int dim = 1 << nQubits;
    // Exact statevector engine: avoids the QUnit separability rounding that
    // QINTERFACE_OPTIMAL applies to generic dense unitaries.
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, nQubits, ZERO_BCI);
    std::vector<complex> in(dim);
    for (int i = 0; i < dim; ++i) {
        in[i] = complex(static_cast<real1>(psi(i).real()),
                        static_cast<real1>(psi(i).imag()));
    }
    qReg->SetQuantumState(in.data());
    apply(qReg);
    std::vector<complex> out(dim);
    qReg->GetQuantumState(out.data());
    Eigen::VectorXcd v(dim);
    for (int i = 0; i < dim; ++i) {
        v(i) = std::complex<double>(real(out[i]), imag(out[i]));
    }
    return v;
}

} // namespace

// ---------------------------------------------------------------------------
// KAK decomposition
// ---------------------------------------------------------------------------
TEST(KAKDecomposition, ReconstructsRandomTwoQubitUnitary) {
    for (unsigned seed = 1; seed <= 5; ++seed) {
        Eigen::Matrix4cd u = sampleUnitary(4, seed);
        qsvt::KAKDecomposition kak(u);
        auto [K1, A, K2] = kak.solve();

        EXPECT_LT((K1 * A * K2 - u).norm(), 1e-9) << "seed " << seed;
        EXPECT_TRUE(isUnitary(K1)) << "seed " << seed;
        EXPECT_TRUE(isUnitary(K2)) << "seed " << seed;
        EXPECT_TRUE(isUnitary(A)) << "seed " << seed;

        // K1, K2 must be local (exact tensor products of single-qubit gates).
        auto [a1, b1] = qsvt::TwoQubitSynthesis::factorizeKron(K1);
        auto [a2, b2] = qsvt::TwoQubitSynthesis::factorizeKron(K2);
        EXPECT_LT((Eigen::kroneckerProduct(a1, b1).eval() - K1).norm(), 1e-9)
            << "K1 not local, seed " << seed;
        EXPECT_LT((Eigen::kroneckerProduct(a2, b2).eval() - K2).norm(), 1e-9)
            << "K2 not local, seed " << seed;
    }
}

// ---------------------------------------------------------------------------
// Two-qubit gate synthesis (KAK -> native gates)
// ---------------------------------------------------------------------------
TEST(TwoQubitSynthesis, FactorizesTensorProduct) {
    Eigen::Matrix2cd u = sampleUnitary(2, 3).block(0, 0, 2, 2);
    Eigen::Matrix2cd v = sampleUnitary(2, 9).block(0, 0, 2, 2);
    Eigen::Matrix4cd k = Eigen::kroneckerProduct(u, v).eval();

    auto [a, b] = qsvt::TwoQubitSynthesis::factorizeKron(k);
    EXPECT_LT((Eigen::kroneckerProduct(a, b).eval() - k).norm(), 1e-10);
}

TEST(TwoQubitSynthesis, CircuitMatrixMatchesUnitary) {
    for (unsigned seed = 1; seed <= 5; ++seed) {
        Eigen::Matrix4cd u = sampleUnitary(4, seed);
        qsvt::TwoQubitSynthesis synth(u);
        // The dense model reproduces U exactly (global phase is tracked).
        EXPECT_LT((synth.circuitMatrix() - u).norm(), 1e-9) << "seed " << seed;
    }
}

TEST(TwoQubitSynthesis, RunsOnQrackSimulator) {
    for (unsigned seed = 41; seed <= 43; ++seed) {
        Eigen::Matrix4cd u = sampleUnitary(4, seed);
        qsvt::TwoQubitSynthesis synth(u);

        // A freshly created register carries an arbitrary global phase, so we
        // use one instance per check and verify the circuit reproduces U |psi>
        // up to a global phase on several (deterministic) input states.
        for (unsigned trial = 0; trial < 4; ++trial) {
            const Eigen::Vector4cd psi = sampleUnitary(4, seed * 10 + trial).col(0);

            QInterfacePtr qReg =
                CreateQuantumInterface(QINTERFACE_CPU, 2, ZERO_BCI);
            std::vector<complex> in(4);
            for (int i = 0; i < 4; ++i) {
                in[i] = complex(static_cast<real1>(psi(i).real()),
                                static_cast<real1>(psi(i).imag()));
            }
            qReg->SetQuantumState(in.data());

            synth.applyTo(qReg, 0, 1);

            std::vector<complex> out(4);
            qReg->GetQuantumState(out.data());
            Eigen::Vector4cd vout;
            for (int i = 0; i < 4; ++i) {
                vout(i) = std::complex<double>(real(out[i]), imag(out[i]));
            }

            const Eigen::Vector4cd expected = u * psi;
            // Single-precision Qrack: use a loose tolerance.
            EXPECT_TRUE(equalUpToGlobalPhase(vout, expected, 2e-3))
                << "seed " << seed << " trial " << trial;
        }
    }
}

// ---------------------------------------------------------------------------
// Full QSVT pipeline
// ---------------------------------------------------------------------------
TEST(QsvtPipeline, BlockEqualsPolynomialOfEigenvalues) {
    auto target = [](double x) { return 0.4 * chebT(1, x) + 0.5 * chebT(3, x); };
    const auto phases = qsvt::QspAngleSolver(3).solve(target).phases;

    for (Eigen::Index k : {2, 4}) {
        const qsvt::Matrix A = sampleHermitian(k, 60u + static_cast<unsigned>(k), 0.9);
        const qsvt::Matrix block = qsvt::qsvtBlock(A, phases);

        // Expected: P(A) = V diag(P(lambda_i)) V^H, P = full QSP (0,0) polynomial.
        Eigen::SelfAdjointEigenSolver<qsvt::Matrix> es(A);
        const Eigen::VectorXd lam = es.eigenvalues();
        Eigen::VectorXcd Pdiag(k);
        for (Eigen::Index i = 0; i < k; ++i) {
            Pdiag(i) = qsvt::QspAngleSolver::response00(lam(i), phases);
        }
        const qsvt::Matrix expected =
            es.eigenvectors() * Pdiag.asDiagonal() * es.eigenvectors().adjoint();

        EXPECT_LT((block - expected).norm(), 1e-9) << "k=" << k;

        // The real part of the transform equals the target polynomial.
        for (Eigen::Index i = 0; i < k; ++i) {
            EXPECT_NEAR(Pdiag(i).real(), target(lam(i)), 1e-6) << "k=" << k;
        }
    }
}

TEST(QsvtPipeline, RotationBlockEncodingIsUnitaryWithCorrectBlock) {
    const qsvt::Matrix A = sampleHermitian(4, 71, 0.8);
    const qsvt::Matrix Uw = qsvt::rotationBlockEncoding(A);
    EXPECT_TRUE(isUnitary(Uw, 1e-9));
    EXPECT_LT((Uw.topLeftCorner(4, 4) - A).norm(), 1e-12);
}

TEST(QsvtPipeline, CompilesMatrixFunctionToRunnableProgram) {
    // Hermitian A with spectrum in [0.3, 1] (well-conditioned, invertible).
    const Eigen::Index k = 4;
    const qsvt::Matrix U = sampleUnitary(k, 123);
    Eigen::VectorXd ev(k);
    for (Eigen::Index i = 0; i < k; ++i) ev(i) = 0.4 + 0.5 * i / (k - 1);
    const qsvt::Matrix A = U * ev.cast<qsvt::Complex>().asDiagonal() * U.adjoint();

    auto target = [](double x) { return 0.5 * chebT(1, x) + 0.3 * chebT(3, x); };
    const qsvt::QsvtProgram prog = qsvt::compileMatrixFunction(A, target, 3);

    EXPECT_TRUE(prog.converged);
    EXPECT_EQ(prog.numQubits, 3); // 2 system + 1 ancilla
    EXPECT_GT(prog.resources.cnot, 0u);
    EXPECT_EQ(prog.resources.total, prog.circuit.size());

    // The compiled gate circuit reproduces the dense QSVT operator.
    const qsvt::Matrix dense = qsvt::denseCircuit(prog.circuit, prog.numQubits);
    const qsvt::Matrix Uphi = qsvt::qsvtUnitary(A, prog.phases);
    Eigen::Index r = 0, c = 0;
    Uphi.cwiseAbs().maxCoeff(&r, &c);
    std::complex<double> gp = dense(r, c) / Uphi(r, c);
    gp /= std::abs(gp);
    EXPECT_LT((dense - gp * Uphi).norm(), 1e-8);
}

TEST(QsvtPipeline, NativeCircuitMatchesDenseOperator) {
    auto target = [](double x) { return 0.4 * chebT(1, x) + 0.5 * chebT(3, x); };
    const auto phases = qsvt::QspAngleSolver(3).solve(target).phases;

    for (Eigen::Index k : {2, 4}) {
        const qsvt::Matrix A = sampleHermitian(k, 80u + static_cast<unsigned>(k), 0.85);
        const std::vector<qsvt::Gate> gates = qsvt::qsvtCircuit(A, phases);
        const int nq = static_cast<int>(std::lround(std::log2(2.0 * k)));

        const qsvt::Matrix dense = qsvt::denseCircuit(gates, nq);
        const qsvt::Matrix Uphi = qsvt::qsvtUnitary(A, phases);

        // Dense gate model reproduces the QSVT operator (up to global phase).
        Eigen::Index r = 0, c = 0;
        Uphi.cwiseAbs().maxCoeff(&r, &c);
        std::complex<double> ph = dense(r, c) / Uphi(r, c);
        ph /= std::abs(ph);
        EXPECT_LT((dense - ph * Uphi).norm(), 1e-8) << "k=" << k;

        // And it runs on the Qrack simulator.
        Eigen::VectorXcd psi = sampleUnitary(2 * k, 90u + static_cast<unsigned>(k)).col(0);
        std::vector<bitLenInt> qubits(nq);
        for (int i = 0; i < nq; ++i) qubits[i] = static_cast<bitLenInt>(i);
        Eigen::VectorXcd vout =
            simulate(nq, psi, [&](QInterfacePtr q) { qsvt::runCircuit(gates, q, qubits); });
        EXPECT_TRUE(equalUpToGlobalPhase(vout, Eigen::VectorXcd(Uphi * psi), 3e-2)) << "k=" << k;
    }
}

// ---------------------------------------------------------------------------
// Hamiltonian simulation (e^{-iHt}) via QSVT
// ---------------------------------------------------------------------------
TEST(HamiltonianSimulation, ApproximatesTimeEvolution) {
    const Eigen::Index k = 4;
    const qsvt::Matrix H = sampleHermitian(k, 31, 0.8); // ||H|| <= 0.8
    const double t = 3.0;

    const qsvt::HamSimProgram prog = qsvt::compileHamiltonianSimulation(H, t, 21);
    EXPECT_TRUE(prog.converged);
    EXPECT_GT(prog.cosResources.cnot, 0u);
    EXPECT_GT(prog.sinResources.cnot, 0u);

    const qsvt::Matrix approx = qsvt::simulatedEvolution(H, prog);

    // Exact evolution e^{-iHt} = V diag(e^{-i t lambda}) V^H.
    Eigen::SelfAdjointEigenSolver<qsvt::Matrix> es(H);
    Eigen::VectorXcd ph(k);
    const std::complex<double> I(0, 1);
    for (Eigen::Index i = 0; i < k; ++i) ph(i) = std::exp(-I * t * es.eigenvalues()(i));
    const qsvt::Matrix exact =
        es.eigenvectors() * ph.asDiagonal() * es.eigenvectors().adjoint();

    EXPECT_LT((approx - exact).norm(), 1e-3);
    EXPECT_LT((approx.adjoint() * approx - qsvt::Matrix::Identity(k, k)).norm(), 1e-2);
}

// ---------------------------------------------------------------------------
// Eigenvalue thresholding / spectral projector via QSVT
// ---------------------------------------------------------------------------
TEST(EigenvalueThreshold, ProjectsOntoPositiveEigenspace) {
    // H with eigenvalues {-0.7, -0.4, 0.4, 0.8}, all bounded away from 0.
    const Eigen::Index k = 4;
    const qsvt::Matrix U = sampleUnitary(k, 314);
    Eigen::VectorXd ev(k);
    ev << -0.7, -0.4, 0.4, 0.8;
    const qsvt::Matrix H = U * ev.cast<qsvt::Complex>().asDiagonal() * U.adjoint();

    // erf/sign is transcendental, so the fit doesn't reach the 1e-6 "converged"
    // bar at finite degree; the projector accuracy below is what matters.
    const qsvt::ThresholdProgram prog =
        qsvt::compileEigenvalueThreshold(H, /*mu=*/0.0, /*w=*/0.1, /*degree=*/25);
    EXPECT_GT(prog.resources.cnot, 0u);

    const qsvt::Matrix P = qsvt::spectralProjector(H, prog);

    // Exact projector onto eigenvalues > 0.
    Eigen::SelfAdjointEigenSolver<qsvt::Matrix> es(H);
    Eigen::VectorXd mask(k);
    for (Eigen::Index i = 0; i < k; ++i) mask(i) = es.eigenvalues()(i) > 0 ? 1.0 : 0.0;
    const qsvt::Matrix exact =
        es.eigenvectors() * mask.cast<qsvt::Complex>().asDiagonal() * es.eigenvectors().adjoint();

    EXPECT_LT((P - exact).norm(), 2e-2);   // ~5e-3 in practice (erf/degree-limited)
    EXPECT_LT((P * P - P).norm(), 2e-2);   // P is close to idempotent

}

// ---------------------------------------------------------------------------
// Resource estimation + QASM export
// ---------------------------------------------------------------------------
TEST(Resources, CountsMatchCircuit) {
    auto target = [](double x) { return 0.4 * chebT(1, x) + 0.5 * chebT(3, x); };
    const auto phases = qsvt::QspAngleSolver(3).solve(target).phases;
    const qsvt::Matrix A = sampleHermitian(2, 5, 0.8);
    const auto gates = qsvt::qsvtCircuit(A, phases);

    const qsvt::ResourceCounts rc = qsvt::countResources(gates, 2);
    std::size_t cx = 0, sq = 0;
    for (const auto& g : gates) (g.isCnot ? cx : sq)++;
    EXPECT_EQ(rc.cnot, cx);
    EXPECT_EQ(rc.singleQubit, sq);
    EXPECT_EQ(rc.total, gates.size());
    EXPECT_GT(rc.depth, 0u);
    EXPECT_LE(rc.depth, gates.size());
}

TEST(Qasm, RoundTripsToSameUnitary) {
    // Build a circuit, export to QASM, parse it back, and check the rebuilt
    // circuit implements the same unitary (up to global phase).
    auto target = [](double x) { return 0.4 * chebT(1, x) + 0.5 * chebT(3, x); };
    const auto phases = qsvt::QspAngleSolver(3).solve(target).phases;
    const qsvt::Matrix A = sampleHermitian(2, 9, 0.8);
    const auto gates = qsvt::qsvtCircuit(A, phases);
    const int nq = 2;

    const std::string qasm = qsvt::toQasm(gates, nq);
    ASSERT_NE(qasm.find("OPENQASM 2.0;"), std::string::npos);
    ASSERT_NE(qasm.find("qreg q[2];"), std::string::npos);

    std::vector<qsvt::Gate> parsed;
    std::istringstream in(qasm);
    std::string line;
    while (std::getline(in, line)) {
        int a = 0, b = 0;
        double th = 0, ph = 0, lam = 0;
        if (std::sscanf(line.c_str(), "cx q[%d],q[%d];", &a, &b) == 2) {
            parsed.push_back(qsvt::Gate::cnot(a, b));
        } else if (std::sscanf(line.c_str(), "U(%lf,%lf,%lf) q[%d];", &th, &ph,
                               &lam, &a) == 4) {
            const std::complex<double> i(0, 1);
            Eigen::Matrix2cd u;
            u << std::cos(th / 2), -std::exp(i * lam) * std::sin(th / 2),
                 std::exp(i * ph) * std::sin(th / 2),
                 std::exp(i * (ph + lam)) * std::cos(th / 2);
            parsed.push_back(qsvt::Gate::single(u, a));
        }
    }
    ASSERT_EQ(parsed.size(), gates.size());

    const qsvt::Matrix orig = qsvt::denseCircuit(gates, nq);
    const qsvt::Matrix rebuilt = qsvt::denseCircuit(parsed, nq);
    Eigen::Index r = 0, c = 0;
    orig.cwiseAbs().maxCoeff(&r, &c);
    std::complex<double> gp = rebuilt(r, c) / orig(r, c);
    gp /= std::abs(gp);
    EXPECT_LT((rebuilt - gp * orig).norm(), 1e-6);
}

// ---------------------------------------------------------------------------
// QSP angle finding
// ---------------------------------------------------------------------------
TEST(QspAngleSolver, FindsPhasesForTargetPolynomials) {
    struct Case { int d; std::function<double(double)> f; };
    const std::vector<Case> cases = {
        {1, [](double x) { return 0.6 * x; }},
        {2, [](double x) { return 0.3 * chebT(0, x) + 0.5 * chebT(2, x); }},
        {3, [](double x) { return 0.4 * chebT(1, x) + 0.5 * chebT(3, x); }},
        {4, [](double x) { return 0.2 * chebT(0, x) + 0.3 * chebT(2, x) + 0.4 * chebT(4, x); }},
        {5, [](double x) { return 0.5 * chebT(1, x) + 0.3 * chebT(3, x) + 0.1 * chebT(5, x); }},
    };
    for (const auto& c : cases) {
        qsvt::QspAngleSolver solver(c.d);
        const qsvt::QspSolveResult res = solver.solve(c.f);
        EXPECT_EQ(static_cast<int>(res.phases.size()), c.d + 1) << "d=" << c.d;
        EXPECT_TRUE(res.converged) << "d=" << c.d << " residual=" << res.residual;
        EXPECT_LT(res.residual, 1e-6) << "d=" << c.d;
    }
}

TEST(QspAngleSolver, ConvergesAtHighDegreeAndNearBoundary) {
    // Homotopy continuation reaches degrees and near-|f|=1 targets where a
    // cold-start Levenberg-Marquardt stalls.
    const auto r1 = qsvt::QspAngleSolver(15).solve(
        [](double x) { return 0.9 * chebT(15, x); });
    EXPECT_TRUE(r1.converged);
    EXPECT_LT(r1.residual, 1e-6);

    const auto r2 = qsvt::QspAngleSolver(51).solve(
        [](double x) { return 0.8 * chebT(51, x); });
    EXPECT_TRUE(r2.converged);
    EXPECT_LT(r2.residual, 1e-6);

    const auto r3 = qsvt::QspAngleSolver(11).solve(
        [](double x) { return 0.97 * chebT(11, x); }); // near boundary
    EXPECT_TRUE(r3.converged);
    EXPECT_LT(r3.residual, 1e-6);
}

TEST(QspAngleSolver, PhasesAreSymmetric) {
    qsvt::QspAngleSolver solver(4);
    const auto res = solver.solve(
        [](double x) { return 0.2 * chebT(0, x) + 0.3 * chebT(2, x) + 0.4 * chebT(4, x); });
    ASSERT_TRUE(res.converged);
    for (int j = 0; j <= 4; ++j) {
        EXPECT_NEAR(res.phases[j], res.phases[4 - j], 1e-9);
    }
}

TEST(QspAngleSolver, DrivesQspCircuitOnSimulator) {
    const int d = 3;
    auto f = [](double x) { return 0.4 * chebT(1, x) + 0.5 * chebT(3, x); };
    const auto res = qsvt::QspAngleSolver(d).solve(f);
    ASSERT_TRUE(res.converged);

    const double x = 0.37;
    const Eigen::Matrix2cd U = qsvt::QspAngleSolver::unitary(x, res.phases);
    const Eigen::Vector2cd expected = U.col(0); // U |0>

    const double s = std::sqrt(1.0 - x * x);
    qsvt::Matrix W(2, 2);
    W << std::complex<double>(x, 0), std::complex<double>(0, s),
         std::complex<double>(0, s), std::complex<double>(x, 0);

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, 1, ZERO_BCI);
    qsvt::Qsp qsp(qReg, qsvt::toQMatrix(W), res.phases);
    qsp.apply();

    std::vector<complex> out(2);
    qReg->GetQuantumState(out.data());
    Eigen::VectorXcd vout(2);
    vout << std::complex<double>(real(out[0]), imag(out[0])),
            std::complex<double>(real(out[1]), imag(out[1]));

    EXPECT_TRUE(equalUpToGlobalPhase(vout, Eigen::VectorXcd(expected), 2e-3));
}

// ---------------------------------------------------------------------------
// Quantum Shannon Decomposition (arbitrary n-qubit synthesis)
// ---------------------------------------------------------------------------
TEST(ShannonDecomposition, CircuitMatrixMatchesUnitary) {
    for (int n = 1; n <= 4; ++n) {
        const Eigen::Index dim = Eigen::Index(1) << n;
        qsvt::Matrix u = sampleUnitary(dim, static_cast<unsigned>(n) + 200);
        qsvt::ShannonDecomposition synth(u);
        EXPECT_LT((synth.circuitMatrix() - u).norm(), 1e-9)
            << "n=" << n << " gates=" << synth.gates().size();
    }
}

TEST(ShannonDecomposition, RunsOnQrackSimulator) {
    for (int n = 2; n <= 4; ++n) {
        const Eigen::Index dim = Eigen::Index(1) << n;
        qsvt::Matrix u = sampleUnitary(dim, static_cast<unsigned>(n) + 300);
        qsvt::ShannonDecomposition synth(u);
        std::vector<bitLenInt> qubits(n);
        for (int i = 0; i < n; ++i) qubits[i] = static_cast<bitLenInt>(i);

        for (unsigned trial = 0; trial < 3; ++trial) {
            Eigen::VectorXcd psi =
                sampleUnitary(dim, static_cast<unsigned>(n) * 100 + trial).col(0);
            Eigen::VectorXcd vout = simulate(n, psi,
                [&](QInterfacePtr q) { synth.applyTo(q, qubits); });
            Eigen::VectorXcd expected = u * psi;
            // Single precision over hundreds of gates (n=4,5): a generous
            // wiring tolerance (a wrong circuit is off by O(1)).
            EXPECT_TRUE(equalUpToGlobalPhase(vout, expected, 3e-2))
                << "n=" << n << " trial=" << trial;
        }
    }
}

// ---------------------------------------------------------------------------
// Cosine-Sine decomposition
// ---------------------------------------------------------------------------
TEST(CSDecomposition, ReconstructsRandomUnitaries) {
    for (Eigen::Index dim : {2, 4, 6, 8}) {
        qsvt::Matrix u = sampleUnitary(dim, static_cast<unsigned>(dim) + 100);
        qsvt::CSDecomposition cs(u);
        auto r = cs.solve();

        EXPECT_LT((qsvt::CSDecomposition::reconstruct(r) - u).norm(), 1e-9)
            << "dim " << dim;
        EXPECT_TRUE(isUnitary(r.L1)) << "dim " << dim;
        EXPECT_TRUE(isUnitary(r.L2)) << "dim " << dim;
        EXPECT_TRUE(isUnitary(r.R1)) << "dim " << dim;
        EXPECT_TRUE(isUnitary(r.R2)) << "dim " << dim;
        for (Eigen::Index k = 0; k < r.theta.size(); ++k) {
            EXPECT_GE(r.theta(k), -1e-12);
            EXPECT_LE(r.theta(k), M_PI / 2 + 1e-12);
        }
    }
}

TEST(CSDecomposition, HandlesIdentityAndBlockDiagonal) {
    // Identity: all cosines = 1, all sines = 0 (exercises the completion path).
    qsvt::Matrix id = qsvt::Matrix::Identity(6, 6);
    auto rId = qsvt::CSDecomposition(id).solve();
    EXPECT_LT((qsvt::CSDecomposition::reconstruct(rId) - id).norm(), 1e-9);

    // Block-diagonal unitary => off-diagonal blocks zero => sines = 0.
    qsvt::Matrix blk = qsvt::Matrix::Zero(6, 6);
    blk.topLeftCorner(3, 3)     = sampleUnitary(3, 21);
    blk.bottomRightCorner(3, 3) = sampleUnitary(3, 22);
    auto rBlk = qsvt::CSDecomposition(blk).solve();
    EXPECT_LT((qsvt::CSDecomposition::reconstruct(rBlk) - blk).norm(), 1e-9);
}

TEST(CSDecomposition, RejectsOddDimension) {
    qsvt::Matrix u = qsvt::Matrix::Identity(3, 3);
    EXPECT_THROW(qsvt::CSDecomposition{u}, qsvt::CSDecompositionException);
}

// ---------------------------------------------------------------------------
// LCU
// ---------------------------------------------------------------------------
TEST(Lcu, ReconstructsMatrixFromPauliStrings) {
    for (Eigen::Index dim : {2, 4}) {
        qsvt::Matrix a(dim, dim);
        for (Eigen::Index r = 0; r < dim; ++r) {
            for (Eigen::Index c = 0; c < dim; ++c) {
                a(r, c) = qsvt::Complex(0.1 * (r + 1), -0.05 * (c + 1));
            }
        }
        qsvt::Lcu lcu(a);
        lcu.generate_pauli_strings();
        lcu.generate_coefs();
        EXPECT_LT((lcu.reconstruct() - a).norm(), 1e-9) << "dim " << dim;
    }
}

// ---------------------------------------------------------------------------
// Block encoding
// ---------------------------------------------------------------------------
TEST(BlockEncoding, ScalarProducesUnitaryWithCorrectBlock) {
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 1, ZERO_BCI);
    qsvt::BlockEncoding be(qReg, 0.6);
    ASSERT_EQ(be.dimension(), 2);

    const auto& u = be.unitary();
    Eigen::Matrix2cd m;
    m << static_cast<std::complex<double>>(u(0, 0)),
         static_cast<std::complex<double>>(u(0, 1)),
         static_cast<std::complex<double>>(u(1, 0)),
         static_cast<std::complex<double>>(u(1, 1));
    EXPECT_NEAR((m.adjoint() * m - Eigen::Matrix2cd::Identity()).norm(), 0.0, 1e-5);
    EXPECT_NEAR(m(0, 0).real(), 0.6, 1e-5);
}

TEST(BlockEncoding, ScalarOutOfRangeThrows) {
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 1, ZERO_BCI);
    EXPECT_THROW(qsvt::BlockEncoding(qReg, 1.5), qsvt::BlockEncodingException);
    EXPECT_THROW(qsvt::BlockEncoding(qReg, -1.5), qsvt::BlockEncodingException);
}

TEST(BlockEncoding, MatrixDilationIsUnitaryWithCorrectTopBlock) {
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 2, ZERO_BCI);
    // A contraction: half of a (unitary) sampled matrix has norm 0.5 < 1.
    qsvt::Matrix a = 0.5 * sampleUnitary(2, 33);
    qsvt::BlockEncoding be(qReg, a);
    ASSERT_EQ(be.dimension(), 4);

    const auto& uq = be.unitary();
    qsvt::Matrix u(4, 4);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            u(r, c) = static_cast<std::complex<double>>(uq(r, c));

    EXPECT_NEAR((u.adjoint() * u - qsvt::Matrix::Identity(4, 4)).norm(), 0.0, 1e-5);
    EXPECT_NEAR((u.topLeftCorner(2, 2) - a).norm(), 0.0, 1e-5);
}

TEST(BlockEncoding, MultiQubitApplyRunsOnSimulator) {
    // A 2x2 contraction yields a 4x4 (2-qubit) block-encoding unitary.
    qsvt::Matrix a = 0.5 * sampleUnitary(2, 55);
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, 2, ZERO_BCI);
    qsvt::BlockEncoding be(qReg, a);
    ASSERT_EQ(be.dimension(), 4);

    Eigen::Matrix4cd U;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            U(r, c) = static_cast<std::complex<double>>(be.unitary()(r, c));

    Eigen::VectorXcd psi = sampleUnitary(4, 88).col(0);
    std::vector<complex> in(4);
    for (int i = 0; i < 4; ++i)
        in[i] = complex(static_cast<real1>(psi(i).real()),
                        static_cast<real1>(psi(i).imag()));
    qReg->SetQuantumState(in.data());

    be.apply(std::vector<bitLenInt>{0, 1}); // compile-and-run the encoding unitary

    std::vector<complex> out(4);
    qReg->GetQuantumState(out.data());
    Eigen::VectorXcd vout(4);
    for (int i = 0; i < 4; ++i)
        vout(i) = std::complex<double>(real(out[i]), imag(out[i]));

    EXPECT_TRUE(equalUpToGlobalPhase(vout, Eigen::VectorXcd(U * psi), 5e-3));
}

// ---------------------------------------------------------------------------
// QSVT engine: ancilla is actually allocated
// ---------------------------------------------------------------------------
TEST(QSVT, AllocatesAncillaQubit) {
    qsvt::QSVT engine(2);
    EXPECT_EQ(engine.ancilla_index(), 2);
    // Ancilla index must be a valid qubit (3 qubits total: 0,1,2).
    EXPECT_NO_THROW(engine.get_simulator()->Prob(engine.ancilla_index()));
}

// ---------------------------------------------------------------------------
// Sanity: raw Qrack gates still behave (kept from the original suite)
// ---------------------------------------------------------------------------
TEST(QrackGates, XGateSetsProbToOne) {
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 2, ZERO_BCI);
    qReg->X(0);
    // Qrack is single precision: don't demand bit-exact equality.
    EXPECT_NEAR(qReg->Prob(0), 1.0, 1e-5);
}

TEST(QrackMeasure, MeasureReturnsCorrectResult) {
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 2, ZERO_BCI);
    qReg->X(0);
    EXPECT_TRUE(qReg->M(0));
}
