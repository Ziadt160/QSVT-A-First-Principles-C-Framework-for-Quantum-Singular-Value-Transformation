#include "SymQspAngleSolver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/FFT>

#include "QspAngleSolver.hpp" // QspAngleSolver::response for residual grading

namespace qsvt {

namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::RowVector3d;
using Eigen::Vector3d;
using Eigen::VectorXd;

constexpr double kPi = 3.14159265358979323846;

// Newton convergence criterion and iteration cap, matching the validated dev
// harness.
constexpr double kNewtonCrit = 1e-12;
constexpr int kNewtonMaxIter = 100;

Matrix3d Rz2(double phi)
{
    const double c = std::cos(2 * phi), s = std::sin(2 * phi);
    Matrix3d m;
    m << c, -s, 0, s, c, 0, 0, 0, 1;
    return m;
}

// y of length n+1: first n are d(response)/d(reduced_k), last is the response.
VectorXd jacComponents(double a, const VectorXd& red, int parity)
{
    const int n = static_cast<int>(red.size());
    const double t = std::acos(std::max(-1.0, std::min(1.0, a)));
    const double c2 = std::cos(2 * t), s2 = std::sin(2 * t);
    Matrix3d B;
    B << c2, 0, -s2, 0, 1, 0, s2, 0, c2;

    std::vector<RowVector3d> L(n);
    L[n - 1] << 0, 1, 0;
    for (int k = n - 2; k >= 0; --k)
        L[k] = L[k + 1] * Rz2(red[k + 1]) * B;

    std::vector<Vector3d> R(n);
    if (parity == 0)
        R[0] << 1, 0, 0;
    else
        R[0] << std::cos(t), 0, std::sin(t);
    for (int k = 1; k < n; ++k)
        R[k] = B * (Rz2(red[k - 1]) * R[k - 1]);

    VectorXd y(n + 1);
    for (int k = 0; k < n; ++k) {
        const double ph = 2 * red[k];
        Matrix3d dRz;
        dRz << -std::sin(ph), -std::cos(ph), 0, std::cos(ph), -std::sin(ph), 0, 0, 0, 0;
        y[k] = 2.0 * (L[k] * dRz * R[k])(0, 0);
    }
    y[n] = (L[n - 1] * Rz2(red[n - 1]) * R[n - 1])(0, 0);
    return y;
}

// Returns (F, dF): achieved reduced Chebyshev coeffs and Jacobian wrt reduced.
//
// The pragma guard silences a spurious GCC -Walloc-size-larger-than that fires
// only under -O3 -march=native: value-range analysis through Eigen's FFT cannot
// prove the `col` vector size (2*dd = 4*n, always small) is bounded, and warns
// about a phantom huge allocation. It is a false positive -- the size is exact.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif
void genJacobian(const VectorXd& red, int parity, VectorXd& F, MatrixXd& dF)
{
    const int n = static_cast<int>(red.size());
    const int dd = 2 * n;
    MatrixXd M = MatrixXd::Zero(2 * dd, n + 1);
    for (int r = 0; r <= n; ++r) {
        const double theta = r * kPi / dd;
        M.row(r) = jacComponents(std::cos(theta), red, parity).transpose();
    }
    const double sgn = (parity % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < n; ++i)
        M.row(n + 1 + i) = sgn * M.row(n - 1 - i);
    for (int i = 0; i <= 2 * n - 2; ++i)
        M.row(2 * n + 1 + i) = M.row(2 * n - 1 - i);

    Eigen::FFT<double> fft;
    MatrixXd Mr(2 * n + 1, n + 1);
    std::vector<double> col(2 * dd);
    std::vector<std::complex<double>> out;
    for (int c = 0; c <= n; ++c) {
        for (int r = 0; r < 2 * dd; ++r) col[r] = M(r, c);
        fft.fwd(out, col);
        for (int r = 0; r <= 2 * n; ++r) Mr(r, c) = out[r].real();
    }
    for (int r = 1; r <= 2 * n - 1; ++r) Mr.row(r) *= 2.0;
    Mr /= static_cast<double>(2 * dd);

    F.resize(n);
    dF.resize(n, n);
    for (int i = 0; i < n; ++i) {
        const int row = parity + 2 * i;
        F[i] = Mr(row, n);
        for (int j = 0; j < n; ++j) dF(i, j) = Mr(row, j);
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

VectorXd newtonSolve(const VectorXd& coef, int parity, double crit, int maxiter, int& iters)
{
    VectorXd red = coef / 2.0;
    VectorXd F;
    MatrixXd dF;
    iters = 0;
    for (int it = 0; it < maxiter; ++it) {
        genJacobian(red, parity, F, dF);
        const VectorXd res = F - coef;
        const double err = res.lpNorm<1>();
        red -= dF.partialPivLu().solve(res);
        ++iters;
        if (err < crit) break;
    }
    return red;
}

std::vector<double> reconstructFull(const VectorXd& red, int parity)
{
    const int n = static_cast<int>(red.size());
    std::vector<double> full;
    if (parity == 1) {
        for (int k = n - 1; k >= 0; --k) full.push_back(red[k]);
        for (int k = 0; k < n; ++k) full.push_back(red[k]);
    } else {
        if (n == 1) {
            full.push_back(2 * red[0]);
        } else {
            for (int k = n - 1; k >= 1; --k) full.push_back(red[k]);
            full.push_back(2 * red[0]);
            for (int k = 1; k < n; ++k) full.push_back(red[k]);
        }
    }
    // Im-protocol -> Re-protocol adapter.
    full.front() -= kPi / 4;
    full.back() -= kPi / 4;
    return full;
}

// Reduced Chebyshev coeffs of a degree-d target callable (parity d%2).
VectorXd chebReduced(const std::function<double(double)>& f, int d)
{
    const int N = d + 1, parity = d % 2, n = d / 2 + 1;
    std::vector<double> c(d + 1, 0.0);
    for (int k = 0; k <= d; ++k) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            const double th = kPi * (j + 0.5) / N;
            sum += f(std::cos(th)) * std::cos(k * th);
        }
        c[k] = (k == 0 ? 1.0 / N : 2.0 / N) * sum;
    }
    VectorXd red(n);
    for (int i = 0; i < n; ++i) red[i] = c[parity + 2 * i];
    return red;
}

} // namespace

QspSolveResult SymQspAngleSolver::solve(
    const std::function<double(double)>& target) const
{
    const int d = degree_;
    const int parity = d % 2;

    int iters = 0;
    const VectorXd coef = chebReduced(target, d);
    const VectorXd red = newtonSolve(coef, parity, kNewtonCrit, kNewtonMaxIter, iters);

    QspSolveResult result;
    result.phases = reconstructFull(red, parity);
    result.iterations = iters;

    // Report the worst-case error over a fine grid on [-1, 1], measured with the
    // shared QspAngleSolver::response (Re<0|U|0>), so the residual is directly
    // comparable to the homotopy solver's.
    double worst = 0.0;
    const int grid = 401;
    for (int i = 0; i < grid; ++i) {
        const double x = -1.0 + 2.0 * i / (grid - 1);
        worst = std::max(
            worst, std::abs(QspAngleSolver::response(x, result.phases) - target(x)));
    }
    result.residual = worst;
    result.converged = worst < 1e-6;
    return result;
}

} // namespace qsvt
