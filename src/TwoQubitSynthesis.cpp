#include "TwoQubitSynthesis.hpp"

#include <cmath>
#include <complex>

#include <eigen3/Eigen/SVD>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include "KAKDecomposition.hpp"

namespace qsvt {

namespace {

using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::Vector4cd;

const std::complex<double> I(0.0, 1.0);

Matrix2cd identity2() { return Matrix2cd::Identity(); }

Matrix2cd sGate()
{
    Matrix2cd s;
    s << 1, 0,
         0, I;
    return s;
}

// Rx(theta) = exp(-i theta X / 2).
Matrix2cd rx(double theta)
{
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    Matrix2cd m;
    m << c, -I * s,
         -I * s, c;
    return m;
}

// Rz(lambda) = diag(e^{-i lambda/2}, e^{+i lambda/2}).
Matrix2cd rz(double lambda)
{
    Matrix2cd m = Matrix2cd::Zero();
    m(0, 0) = std::exp(-0.5 * I * lambda);
    m(1, 1) = std::exp(0.5 * I * lambda);
    return m;
}

// Two-qubit Pauli products in the computational basis, qubit-0 = low bit.
Matrix4cd pauliProduct(char which)
{
    Matrix2cd p;
    switch (which) {
    case 'X': p << 0, 1, 1, 0; break;
    case 'Y': p << 0, -I, I, 0; break;
    default:  p << 1, 0, 0, -1; break; // 'Z'
    }
    return Eigen::kroneckerProduct(p, p).eval();
}

Matrix4cd magicBasis()
{
    Matrix4cd Q;
    Q << 1, 0, 0, I,
         0, I, 1, 0,
         0, I, -1, 0,
         1, 0, 0, -I;
    return Q / std::sqrt(2.0);
}

// Append the 4-CNOT canonical entangler exp(i(ax XX + ay YY + az ZZ)) on
// (qLow, qHigh). The XX and ZZ rotations are realised together as
// CNOT (Rx(-2ax) (x) Rz(-2az)) CNOT (the two inner CNOTs of the separate
// exp(i ax XX) and exp(i az ZZ) circuits cancel); YY adds the other two CNOTs.
void appendCanonical(std::vector<Gate>& g, double ax, double ay, double az,
                     int qLow, int qHigh)
{
    // exp(i(ax XX + az ZZ)) -- 2 CNOTs.
    g.push_back(Gate::cnot(qLow, qHigh));
    g.push_back(Gate::single(rx(-2.0 * ax), qLow));
    g.push_back(Gate::single(rz(-2.0 * az), qHigh));
    g.push_back(Gate::cnot(qLow, qHigh));

    // exp(i ay YY) = (S (x) S) CNOT (Rx(-2ay) (x) I) CNOT (S^H (x) S^H) -- 2 CNOTs.
    const Matrix2cd Sd = sGate().adjoint();
    g.push_back(Gate::single(Sd, qLow));
    g.push_back(Gate::single(Sd, qHigh));
    g.push_back(Gate::cnot(qLow, qHigh));
    g.push_back(Gate::single(rx(-2.0 * ay), qLow));
    g.push_back(Gate::cnot(qLow, qHigh));
    g.push_back(Gate::single(sGate(), qLow));
    g.push_back(Gate::single(sGate(), qHigh));
}

} // namespace

std::pair<Matrix2cd, Matrix2cd>
TwoQubitSynthesis::factorizeKron(const Matrix4cd& K)
{
    // Van Loan-Pitsianis rearrangement: R[i*2+j, k*2+l] = K[i*2+k, j*2+l] is
    // rank-1 (= vec(A) vec(B)^T) exactly when K = A (x) B.
    Matrix4cd R;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    R(i * 2 + j, k * 2 + l) = K(i * 2 + k, j * 2 + l);

    Eigen::JacobiSVD<Matrix4cd> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const double s = svd.singularValues()(0);
    const Vector4cd u = svd.matrixU().col(0);
    const Vector4cd v = svd.matrixV().col(0); // R = U S V^H
    const double rs = std::sqrt(s);

    Matrix2cd A, B;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            A(i, j) = rs * u(i * 2 + j);
    for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 2; ++l)
            B(k, l) = rs * std::conj(v(k * 2 + l));
    return {A, B};
}

std::vector<Gate> synthesizeTwoQubit(const Matrix4cd& U, int qLow, int qHigh)
{
    std::vector<Gate> g;

    KAKDecomposition kak(U);
    auto [K1, A, K2] = kak.solve();

    // Apply K2 first (U |psi> = K1 A K2 |psi>); its single-qubit factors.
    auto [u2, v2] = TwoQubitSynthesis::factorizeKron(K2);
    g.push_back(Gate::single(v2, qLow));
    g.push_back(Gate::single(u2, qHigh));

    // Recover (gamma, ax, ay, az) from A, which is diagonal in the magic basis,
    // by matching its phases against the magic-basis eigenvalue signs of
    // XX/YY/ZZ (convention-independent, so the circuit reproduces A exactly).
    const Matrix4cd Q = magicBasis();
    const Vector4cd diag = (Q.adjoint() * A * Q).diagonal();
    Eigen::Vector4d phi;
    for (int k = 0; k < 4; ++k) {
        phi(k) = std::arg(diag(k));
    }
    const Matrix4cd Xm = Q.adjoint() * pauliProduct('X') * Q;
    const Matrix4cd Ym = Q.adjoint() * pauliProduct('Y') * Q;
    const Matrix4cd Zm = Q.adjoint() * pauliProduct('Z') * Q;
    Eigen::Matrix4d S; // columns: [1, sx, sy, sz]
    for (int k = 0; k < 4; ++k) {
        S(k, 0) = 1.0;
        S(k, 1) = Xm(k, k).real();
        S(k, 2) = Ym(k, k).real();
        S(k, 3) = Zm(k, k).real();
    }
    const Eigen::Vector4d sol = S.colPivHouseholderQr().solve(phi);
    const double gamma = sol(0), ax = sol(1), ay = sol(2), az = sol(3);

    appendCanonical(g, ax, ay, az, qLow, qHigh);

    // Global phase e^{i gamma} (as a single-qubit gate e^{i gamma} I).
    g.push_back(Gate::single(std::exp(I * gamma) * identity2(), qLow));

    // Finally apply K1.
    auto [u1, v1] = TwoQubitSynthesis::factorizeKron(K1);
    g.push_back(Gate::single(v1, qLow));
    g.push_back(Gate::single(u1, qHigh));

    return g;
}

TwoQubitSynthesis::TwoQubitSynthesis(const Matrix4cd& U)
    : U_(U)
{
    build();
}

void TwoQubitSynthesis::build()
{
    gates_ = synthesizeTwoQubit(U_, 0, 1);
}

Matrix4cd TwoQubitSynthesis::circuitMatrix() const
{
    return Matrix4cd(denseCircuit(gates_, 2));
}

void TwoQubitSynthesis::applyTo(Qrack::QInterfacePtr qReg, bitLenInt q0,
                                bitLenInt q1) const
{
    runCircuit(gates_, qReg, {q0, q1});
}

} // namespace qsvt
