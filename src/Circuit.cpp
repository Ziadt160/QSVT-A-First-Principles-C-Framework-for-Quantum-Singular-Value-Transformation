#include "Gate.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <sstream>
#include <vector>

namespace qsvt {

namespace {

// Embed a single-qubit gate acting on logical qubit q into the 2^n space
// (qubit 0 = least significant bit of the basis index).
Matrix embedSingle(const Eigen::Matrix2cd& m, int q, int n)
{
    const Eigen::Index dim = Eigen::Index(1) << n;
    Matrix full = Matrix::Zero(dim, dim);
    for (Eigen::Index i = 0; i < dim; ++i) {
        const int bq = static_cast<int>((i >> q) & 1);
        const Eigen::Index cleared = i & ~(Eigen::Index(1) << q);
        for (int bp = 0; bp < 2; ++bp) {
            const Eigen::Index row = cleared | (Eigen::Index(bp) << q);
            full(row, i) = m(bp, bq);
        }
    }
    return full;
}

// Embed a CNOT(control -> target) into the 2^n space.
Matrix embedCnot(int control, int target, int n)
{
    const Eigen::Index dim = Eigen::Index(1) << n;
    Matrix full = Matrix::Zero(dim, dim);
    for (Eigen::Index i = 0; i < dim; ++i) {
        Eigen::Index out = i;
        if ((i >> control) & 1) {
            out ^= (Eigen::Index(1) << target);
        }
        full(out, i) = 1.0;
    }
    return full;
}

} // namespace

Matrix denseCircuit(const std::vector<Gate>& gates, int nQubits)
{
    const Eigen::Index dim = Eigen::Index(1) << nQubits;
    Matrix result = Matrix::Identity(dim, dim);
    for (const Gate& g : gates) {
        const Matrix m = g.isCnot ? embedCnot(g.control, g.target, nQubits)
                                  : embedSingle(g.matrix, g.target, nQubits);
        result = m * result; // first gate is applied first
    }
    return result;
}

void runCircuit(const std::vector<Gate>& gates, Qrack::QInterfacePtr qReg,
                const std::vector<bitLenInt>& qubits)
{
    for (const Gate& g : gates) {
        if (g.isCnot) {
            qReg->CNOT(qubits[g.control], qubits[g.target]);
        } else {
            const QMatrix m = toQMatrix(Matrix(g.matrix));
            qReg->Mtrx(m.data(), qubits[g.target]);
        }
    }
}

namespace {

bool touches(const Gate& g, int q)
{
    return g.target == q || (g.isCnot && g.control == q);
}

// Index of the first gate after `from` that touches qubit `q`, or -1.
int nextTouching(const std::vector<Gate>& g, std::size_t from, int q)
{
    for (std::size_t j = from; j < g.size(); ++j) {
        if (touches(g[j], q)) {
            return static_cast<int>(j);
        }
    }
    return -1;
}

} // namespace

ResourceCounts countResources(const std::vector<Gate>& gates, int nQubits)
{
    ResourceCounts r;
    std::vector<std::size_t> layer(static_cast<std::size_t>(nQubits), 0);
    for (const Gate& g : gates) {
        ++r.total;
        if (g.isCnot) {
            ++r.cnot;
            const std::size_t t =
                std::max(layer[g.control], layer[g.target]) + 1;
            layer[g.control] = layer[g.target] = t;
        } else {
            ++r.singleQubit;
            layer[g.target] += 1;
        }
    }
    r.depth = layer.empty() ? 0 : *std::max_element(layer.begin(), layer.end());
    return r;
}

namespace {

// Factor a 2x2 unitary into OpenQASM's U(theta, phi, lambda) (= IBM U3), up to
// an unobservable global phase.
void zyzAngles(const Eigen::Matrix2cd& m, double& theta, double& phi,
               double& lambda)
{
    const double m00 = std::abs(m(0, 0));
    const double m10 = std::abs(m(1, 0));
    theta = 2.0 * std::atan2(m10, m00);

    if (m10 < 1e-12) { // diagonal: only phi+lambda matters
        phi = std::arg(m(1, 1)) - std::arg(m(0, 0));
        lambda = 0.0;
    } else if (m00 < 1e-12) { // antidiagonal
        phi = std::arg(m(1, 0)) - std::arg(-m(0, 1));
        lambda = 0.0;
    } else {
        // Extract phi and lambda DIRECTLY relative to arg(M00). Averaging
        // phi+lambda and phi-lambda (each only defined mod 2*pi) is wrong when
        // the two wrap inconsistently -- it shifts phi/lambda by pi and flips
        // the off-diagonal signs. Only e^{i phi}, e^{i lambda} matter, so any
        // representative of each angle is fine.
        phi = std::arg(m(1, 0)) - std::arg(m(0, 0));
        lambda = std::arg(-m(0, 1)) - std::arg(m(0, 0));
    }
}

} // namespace

std::string toQasm(const std::vector<Gate>& gates, int nQubits)
{
    std::ostringstream out;
    out << "OPENQASM 2.0;\n";
    out << "include \"qelib1.inc\";\n";
    out << "qreg q[" << nQubits << "];\n";
    out.setf(std::ios::fixed);
    out.precision(10);
    for (const Gate& g : gates) {
        if (g.isCnot) {
            out << "cx q[" << g.control << "],q[" << g.target << "];\n";
        } else {
            double theta, phi, lambda;
            zyzAngles(g.matrix, theta, phi, lambda);
            out << "U(" << theta << "," << phi << "," << lambda << ") q["
                << g.target << "];\n";
        }
    }
    return out.str();
}

std::vector<Gate> optimizeCircuit(const std::vector<Gate>& gates)
{
    std::vector<Gate> g = gates;
    bool changed = true;
    while (changed) {
        changed = false;
        // One forward sweep; on a reduction, re-examine the current index
        // (without restarting the whole sweep). Cascades that expose earlier
        // adjacencies are caught by the next sweep.
        std::size_t i = 0;
        while (i < g.size()) {
            if (!g[i].isCnot) {
                // Fuse with the next gate on the same qubit, if it is also a
                // single-qubit gate there (a CNOT touching the qubit blocks it).
                const int j = nextTouching(g, i + 1, g[i].target);
                if (j >= 0 && !g[std::size_t(j)].isCnot &&
                    g[std::size_t(j)].target == g[i].target) {
                    g[i].matrix = g[std::size_t(j)].matrix * g[i].matrix;
                    g.erase(g.begin() + j);
                    changed = true;
                    continue; // try to fuse the merged gate again
                }
            } else {
                // Cancel with the next gate touching either qubit, if it is the
                // identical CNOT (nothing in between touches control or target).
                const int jc = nextTouching(g, i + 1, g[i].control);
                const int jt = nextTouching(g, i + 1, g[i].target);
                int j = -1;
                if (jc >= 0 && jt >= 0) j = std::min(jc, jt);
                else j = std::max(jc, jt); // the one that exists, or -1
                if (j >= 0 && g[std::size_t(j)].isCnot &&
                    g[std::size_t(j)].control == g[i].control &&
                    g[std::size_t(j)].target == g[i].target) {
                    g.erase(g.begin() + j);
                    g.erase(g.begin() + i);
                    changed = true;
                    continue; // g[i] is now the following gate; re-examine
                }
            }
            ++i;
        }
    }
    return g;
}

} // namespace qsvt
