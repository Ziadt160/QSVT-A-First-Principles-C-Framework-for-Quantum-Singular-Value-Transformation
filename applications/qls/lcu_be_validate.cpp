// Dense validation of the native-gate LCU block-encoding (LcuBlockEncoding).
//
// For the 1D transverse-field Ising chain H = J sum Z_i Z_{i+1} + h sum X_i
// (same as the Python tfim_terms, J=1, h=0.6) at n_system = 2..6, we:
//   * build LcuBlockEncoding,
//   * extract the all-ancilla-|0> block of the circuit unitary (top-left
//     2^n_system block, since ancilla + work qubits are the HIGH qubits),
//   * check  ||block - A/alpha|| / ||A/alpha|| < 1e-10,
//   * print n, L, ancilla, block_err, cnot_count, and dense 0.58*4^(n+1).
//
// Two block extractors are used and CROSS-CHECKED against each other:
//   (a) full  qsvt::denseCircuit(gates, N) then .topLeftCorner  -- the spec
//       path; O(#gates * 8^N), affordable only for small N;
//   (b) circuitTopLeftBlock(gates, N, n) -- applies the SAME gate list to just
//       the 2^n input columns whose ancilla bits are 0, giving an identical
//       block at O(#gates * 2^N * 2^n). Used for every n; for n <= 4 we also
//       run (a) and assert the two blocks agree to 1e-12, proving (b) faithfully
//       reproduces the denseCircuit result.
//
// Build:
//   g++ -O3 -std=c++17 -Iinclude -I/usr/include/eigen3
//       applications/qls/lcu_be_validate.cpp
//       src/LcuBlockEncoding.cpp src/Circuit.cpp -o /tmp/lcu_be && /tmp/lcu_be

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include "Common.hpp"
#include "Gate.hpp"
#include "LcuBlockEncoding.hpp"

using qsvt::Complex;
using qsvt::Matrix;
using qsvt::PauliTerm;

namespace {

Eigen::Matrix2cd pauli2(char p)
{
    Eigen::Matrix2cd m;
    switch (p) {
    case 'I': m << 1, 0, 0, 1; break;
    case 'X': m << 0, 1, 1, 0; break;
    case 'Y': m << Complex(0, 0), Complex(0, -1), Complex(0, 1), Complex(0, 0); break;
    case 'Z': m << 1, 0, 0, -1; break;
    default: m << 1, 0, 0, 1; break;
    }
    return m;
}

// Dense Pauli string: s[0] is the MOST significant qubit (matches Python
// pauli_string = kron(s[0], s[1], ...)).
Matrix pauliString(const std::string& s)
{
    Matrix m(1, 1);
    m(0, 0) = 1.0;
    for (char ch : s) {
        m = Eigen::kroneckerProduct(m, pauli2(ch)).eval();
    }
    return m;
}

// Apply the gate list to the first 2^n columns of the N-qubit identity and
// return the top-left 2^n x 2^n block. This equals
// denseCircuit(gates, N).topLeftCorner(2^n, 2^n) exactly, but costs only
// O(#gates * 2^N * 2^n) instead of O(#gates * 8^N): we evolve a (2^N x 2^n)
// "state" matrix rather than the full (2^N x 2^N) unitary. Qubit 0 = LSB,
// matching denseCircuit's convention; the 2^n input columns are the basis
// states with all high (ancilla + work) qubits = 0, i.e. column indices
// 0..2^n-1.
Matrix circuitTopLeftBlock(const std::vector<qsvt::Gate>& gates, int N, int n)
{
    const Eigen::Index dim = Eigen::Index(1) << N;
    const Eigen::Index cols = Eigen::Index(1) << n;
    Matrix st = Matrix::Zero(dim, cols);
    for (Eigen::Index c = 0; c < cols; ++c) {
        st(c, c) = 1.0; // |c> with ancilla/work = 0
    }
    for (const qsvt::Gate& g : gates) {
        Matrix next = st; // copy; overwrite touched rows
        if (g.isCnot) {
            const Eigen::Index cb = Eigen::Index(1) << g.control;
            const Eigen::Index tb = Eigen::Index(1) << g.target;
            for (Eigen::Index r = 0; r < dim; ++r) {
                if (r & cb) {
                    next.row(r) = st.row(r ^ tb);
                }
            }
        } else {
            const Eigen::Matrix2cd& M = g.matrix;
            const Eigen::Index tb = Eigen::Index(1) << g.target;
            for (Eigen::Index r = 0; r < dim; ++r) {
                if (r & tb) {
                    continue; // handle each pair once, from the bit=0 row
                }
                const Eigen::Index r1 = r | tb;
                const auto a0 = st.row(r);
                const auto a1 = st.row(r1);
                next.row(r) = M(0, 0) * a0 + M(0, 1) * a1;
                next.row(r1) = M(1, 0) * a0 + M(1, 1) * a1;
            }
        }
        st.swap(next);
    }
    return st.topRows(cols);
}

// 1D TFIM open chain, J=1, h=0.6 (matches Python tfim_terms).
std::vector<PauliTerm> tfimTerms(int n, double J = 1.0, double h = 0.6)
{
    std::vector<PauliTerm> terms;
    for (int i = 0; i < n - 1; ++i) {
        std::string s(static_cast<std::size_t>(n), 'I');
        s[static_cast<std::size_t>(i)] = 'Z';
        s[static_cast<std::size_t>(i + 1)] = 'Z';
        terms.push_back({Complex(J, 0.0), s});
    }
    for (int i = 0; i < n; ++i) {
        std::string s(static_cast<std::size_t>(n), 'I');
        s[static_cast<std::size_t>(i)] = 'X';
        terms.push_back({Complex(h, 0.0), s});
    }
    return terms;
}

} // namespace

int main()
{
    std::printf("%3s %8s %4s %14s %12s %16s %12s\n", "n", "L_terms", "anc",
                "block_err", "cnots", "dense_cnots", "ratio");

    bool allPass = true;
    for (int n = 2; n <= 6; ++n) {
        const std::vector<PauliTerm> terms = tfimTerms(n);
        const int L = static_cast<int>(terms.size());

        qsvt::LcuBlockEncoding be(terms, n);
        const int m = be.numAncilla();
        const double alpha = be.alpha();
        const auto& gates = be.gates();

        const int totalQubits = n + m + ((m >= 2) ? (m - 1) : 0);
        const Eigen::Index dimn = Eigen::Index(1) << n;

        // Efficient, faithful block extractor (used for every n).
        const Matrix block = circuitTopLeftBlock(gates, totalQubits, n);

        // Cross-check against the full denseCircuit path where affordable.
        double crossErr = 0.0;
        if (totalQubits <= 9) {
            const Matrix U = qsvt::denseCircuit(gates, totalQubits);
            const Matrix blockFull = U.topLeftCorner(dimn, dimn);
            crossErr = (block - blockFull).norm();
        }

        // Reference A/alpha.
        Matrix A = Matrix::Zero(dimn, dimn);
        for (const PauliTerm& t : terms) {
            A += t.coeff * pauliString(t.pauli);
        }
        const Matrix target = A / alpha;

        const double err = (block - target).norm() / target.norm();

        const auto rc = qsvt::countResources(gates, totalQubits);
        const double dense = 0.58 * std::pow(4.0, n + 1);

        std::printf("%3d %8d %4d %14.3e %12zu %16.3e %11.1fx\n", n, L, m, err,
                    rc.cnot, dense, dense / static_cast<double>(rc.cnot));
        std::fflush(stdout);

        if (!(err < 1e-10)) {
            allPass = false;
        }
        if (crossErr > 1e-12) {
            std::printf("  CROSS-CHECK FAIL at n=%d: ||block_eff - "
                        "denseCircuit_block|| = %.3e\n",
                        n, crossErr);
            allPass = false;
        }
    }

    std::printf("\n%s\n", allPass ? "ALL PASS (block_err < 1e-10)"
                                  : "FAIL: some block_err >= 1e-10");
    return allPass ? 0 : 1;
}
