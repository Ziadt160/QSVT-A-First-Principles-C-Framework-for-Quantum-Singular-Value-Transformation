#include "ShannonDecomposition.hpp"

#include <cmath>
#include <complex>

#include <eigen3/Eigen/Eigenvalues>

#include "CSDecomposition.hpp"
#include "TwoQubitSynthesis.hpp"

namespace qsvt {

namespace {

const std::complex<double> kI(0.0, 1.0);

Eigen::Matrix2cd ry(double theta)
{
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    Eigen::Matrix2cd m;
    m << c, -s,
         s, c;
    return m;
}

Eigen::Matrix2cd rz(double theta)
{
    Eigen::Matrix2cd m = Eigen::Matrix2cd::Zero();
    m(0, 0) = std::exp(-0.5 * kI * theta);
    m(1, 1) = std::exp(0.5 * kI * theta);
    return m;
}

} // namespace

ShannonDecomposition::ShannonDecomposition(const Matrix& U)
{
    if (U.rows() != U.cols()) {
        throw ShannonException("matrix must be square");
    }
    const Eigen::Index dim = U.rows();
    n_ = static_cast<int>(std::lround(std::log2(static_cast<double>(dim))));
    if (dim == 0 || (Eigen::Index(1) << n_) != dim) {
        throw ShannonException("dimension must be a power of two");
    }

    std::vector<int> qubits(n_);
    for (int i = 0; i < n_; ++i) {
        qubits[i] = i;
    }
    decompose(U, qubits);
    gates_ = optimizeCircuit(gates_);
}

void ShannonDecomposition::appendUCR(char axis, const RealVector& angles,
                                     const std::vector<int>& controls, int target)
{
    const int k = static_cast<int>(controls.size());
    if (k == 0) {
        const Eigen::Matrix2cd g = (axis == 'Y') ? ry(angles(0)) : rz(angles(0));
        gates_.push_back(Gate::single(g, target));
        return;
    }

    // Mottonen flat construction: a uniformly-controlled rotation with k
    // controls realised as exactly 2^k single-qubit rotations interleaved with
    // 2^k CNOTs in Gray-code order (vs the recursive 2^{k+1}-2). The applied
    // angles are theta = (1/2^k) (-1)^{<i, gray(j)>} * angles.
    const int N = 1 << k;
    RealVector theta(N);
    for (int i = 0; i < N; ++i) {
        const int gi = i ^ (i >> 1); // Gray code of i
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            const int sign = (__builtin_popcount(gi & j) & 1) ? -1 : 1;
            sum += sign * angles(j);
        }
        theta(i) = sum / N;
    }

    for (int i = 0; i < N; ++i) {
        const Eigen::Matrix2cd g = (axis == 'Y') ? ry(theta(i)) : rz(theta(i));
        gates_.push_back(Gate::single(g, target));
        // CNOT control: the bit that flips in the Gray-code sequence, i.e. the
        // number of trailing zeros of (i+1); the final one wraps to the MSB.
        const int ctrlBit = (i == N - 1) ? (k - 1) : __builtin_ctz(i + 1);
        gates_.push_back(Gate::cnot(controls[ctrlBit], target));
    }
}

void ShannonDecomposition::appendMultiplexor(const Matrix& V1, const Matrix& V2,
                                             const std::vector<int>& lower,
                                             int msb)
{
    // blockdiag(V1, V2) = (I (x) E) . blockdiag(D, D^H) . (I (x) F), where
    // V1 V2^H = E D^2 E^H and F = D^H E^H V1. The middle factor is a
    // uniformly-controlled Rz on `msb` with angle -2 arg(D_k) per control state.
    const Matrix VV = V1 * V2.adjoint();
    Eigen::ComplexSchur<Matrix> schur(VV); // VV is unitary (=> normal => T diagonal)
    if (schur.info() != Eigen::Success) {
        throw ShannonException("Schur decomposition failed");
    }
    const Matrix E = schur.matrixU();
    const Matrix T = schur.matrixT();

    const Eigen::Index m = VV.rows();
    Vector delta(m);
    RealVector lambda(m);
    for (Eigen::Index k = 0; k < m; ++k) {
        const Complex d = std::sqrt(T(k, k));
        delta(k) = d;
        lambda(k) = -2.0 * std::arg(d);
    }
    const Matrix Dh = delta.conjugate().asDiagonal(); // D^H (diagonal)
    const Matrix F = Dh * E.adjoint() * V1;

    // Apply order: (I (x) F), then UCRz, then (I (x) E).
    decompose(F, lower);
    appendUCR('Z', lambda, lower, msb);
    decompose(E, lower);
}

void ShannonDecomposition::decompose(const Matrix& U,
                                     const std::vector<int>& qubits)
{
    if (qubits.size() == 1) {
        gates_.push_back(Gate::single(Eigen::Matrix2cd(U), qubits[0]));
        return;
    }

    if (qubits.size() == 2) {
        // Optimal-er base case: 4-CNOT two-qubit synthesis instead of recursing
        // (which would cost 6 CNOTs via the generic CSD path).
        const std::vector<Gate> sub =
            synthesizeTwoQubit(Eigen::Matrix4cd(U), qubits[0], qubits[1]);
        gates_.insert(gates_.end(), sub.begin(), sub.end());
        return;
    }

    CSDecomposition cs(U);
    const CSDecomposition::Result r = cs.solve();

    const std::vector<int> lower(qubits.begin(), qubits.end() - 1);
    const int msb = qubits.back();

    // U = blockdiag(L1, L2) . [[C,-S],[S,C]] . blockdiag(R1, R2)^H.
    // Apply right multiplexor (adjoint) first, then the Ry rotation, then left.
    appendMultiplexor(r.R1.adjoint(), r.R2.adjoint(), lower, msb);
    appendUCR('Y', RealVector(2.0 * r.theta), lower, msb);
    appendMultiplexor(r.L1, r.L2, lower, msb);
}

} // namespace qsvt
