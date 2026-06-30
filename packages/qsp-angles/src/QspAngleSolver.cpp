// VENDORED from the QSVT_Project root (src/QspAngleSolver.cpp).
// The ONLY change vs the canonical source is the Eigen include path
// (<eigen3/Eigen/Dense> -> <Eigen/Dense>). Keep in sync with the root.

#include "QspAngleSolver.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace qsvt {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Build the full symmetric phase sequence (length d+1) from the free parameters
// (length floor(d/2)+1): phi_j = free[min(j, d-j)].
std::vector<double> expandSymmetric(const Eigen::VectorXd& free, int d)
{
    std::vector<double> phi(d + 1);
    for (int j = 0; j <= d; ++j) {
        phi[j] = free(std::min(j, d - j));
    }
    return phi;
}

// Response g = Re<0|U(x)|0> together with the exact gradient d g / d phi_k for
// all k, computed in one O(d) sweep via prefix/suffix products of the QSP
// factor list M = [E(phi_0), W, E(phi_1), W, ..., W, E(phi_d)].
void responseGrad(double x, const std::vector<double>& phi, double& g,
                  std::vector<double>& grad)
{
    using cd = std::complex<double>;
    const cd I(0.0, 1.0);
    const int d = static_cast<int>(phi.size()) - 1;
    const double s = std::sqrt(std::max(0.0, 1.0 - x * x));

    Eigen::Matrix2cd W;
    W << cd(x, 0), cd(0, s),
         cd(0, s), cd(x, 0);
    auto E = [&](double p) {
        Eigen::Matrix2cd m = Eigen::Matrix2cd::Zero();
        m(0, 0) = std::exp(I * p);
        m(1, 1) = std::exp(-I * p);
        return m;
    };

    const int n = 2 * d + 1; // number of factors
    std::vector<Eigen::Matrix2cd> M(n);
    M[0] = E(phi[0]);
    for (int k = 1; k <= d; ++k) {
        M[2 * k - 1] = W;
        M[2 * k] = E(phi[k]);
    }

    std::vector<Eigen::Matrix2cd> pre(n + 1), suf(n + 1);
    pre[0] = Eigen::Matrix2cd::Identity();
    for (int j = 0; j < n; ++j) pre[j + 1] = pre[j] * M[j];
    suf[n] = Eigen::Matrix2cd::Identity();
    for (int j = n - 1; j >= 0; --j) suf[j] = M[j] * suf[j + 1];

    g = pre[n](0, 0).real();
    grad.assign(d + 1, 0.0);
    for (int k = 0; k <= d; ++k) {
        // d E(phi_k)/d phi_k = i Z E(phi_k) = diag(i e^{i phi}, -i e^{-i phi}).
        Eigen::Matrix2cd dE = Eigen::Matrix2cd::Zero();
        dE(0, 0) = I * std::exp(I * phi[k]);
        dE(1, 1) = -I * std::exp(-I * phi[k]);
        const Eigen::Matrix2cd D = pre[2 * k] * dE * suf[2 * k + 1];
        grad[k] = D(0, 0).real();
    }
}

} // namespace

Eigen::Matrix2cd QspAngleSolver::unitary(double x,
                                         const std::vector<double>& phases)
{
    using cd = std::complex<double>;
    const cd I(0.0, 1.0);
    const double s = std::sqrt(std::max(0.0, 1.0 - x * x));

    // W(x) = e^{i arccos(x) X}.
    Eigen::Matrix2cd W;
    W << cd(x, 0), cd(0, s),
         cd(0, s), cd(x, 0);

    auto ephiZ = [&](double phi) {
        Eigen::Matrix2cd m = Eigen::Matrix2cd::Zero();
        m(0, 0) = std::exp(I * phi);
        m(1, 1) = std::exp(-I * phi);
        return m;
    };

    Eigen::Matrix2cd U = ephiZ(phases[0]);
    for (std::size_t k = 1; k < phases.size(); ++k) {
        U = U * W * ephiZ(phases[k]);
    }
    return U;
}

QspSolveResult QspAngleSolver::solve(
    const std::function<double(double)>& target) const
{
    const int d = degree_;
    const int nFree = d / 2 + 1;
    const int m = std::max(2 * nFree, d + 2); // overdetermined system

    // Chebyshev-like sample nodes in (0, 1); parity fixes the rest of [-1, 1].
    std::vector<double> xs(m), fx(m), base(m);
    const std::vector<double> zeroPhases(d + 1, 0.0);
    for (int i = 0; i < m; ++i) {
        xs[i] = std::cos(kPi * (i + 0.5) / (2.0 * m));
        fx[i] = target(xs[i]);
        // Phi = 0 produces the Chebyshev polynomial T_d -- a known exact start
        // for the homotopy below.
        base[i] = response(xs[i], zeroPhases);
    }

    // Residual at the nodes for a given free vector.
    auto residual = [&](const Eigen::VectorXd& fr, const std::vector<double>& goal) {
        const std::vector<double> phi = expandSymmetric(fr, d);
        Eigen::VectorXd r(m);
        for (int i = 0; i < m; ++i) r(i) = response(xs[i], phi) - goal[i];
        return r;
    };

    // Residual and exact Jacobian (analytic, folded onto the symmetric params).
    std::vector<double> grad(d + 1);
    auto residualJac = [&](const Eigen::VectorXd& fr, const std::vector<double>& goal,
                           Eigen::VectorXd& r, Eigen::MatrixXd& J) {
        const std::vector<double> phi = expandSymmetric(fr, d);
        J.setZero();
        for (int i = 0; i < m; ++i) {
            double g;
            responseGrad(xs[i], phi, g, grad);
            r(i) = g - goal[i];
            for (int k = 0; k <= d; ++k) J(i, std::min(k, d - k)) += grad[k];
        }
    };

    // One Levenberg-Marquardt solve fitting `goal`, warm-started from `free`.
    auto lmFit = [&](const std::vector<double>& goal, Eigen::VectorXd free) {
        Eigen::VectorXd r(m);
        Eigen::MatrixXd J(m, nFree);
        residualJac(free, goal, r, J);
        double lambda = 1e-3;
        for (int iter = 0; iter < 100 && r.cwiseAbs().maxCoeff() > 1e-13; ++iter) {
            const Eigen::MatrixXd JtJ = J.transpose() * J;
            const Eigen::VectorXd Jtr = J.transpose() * r;
            bool stepped = false;
            for (int tries = 0; tries < 30; ++tries) {
                const Eigen::VectorXd delta =
                    (JtJ + lambda * Eigen::MatrixXd::Identity(nFree, nFree))
                        .ldlt()
                        .solve(-Jtr);
                const Eigen::VectorXd cand = free + delta;
                const Eigen::VectorXd rc = residual(cand, goal);
                if (rc.squaredNorm() < r.squaredNorm()) {
                    free = cand;
                    lambda = std::max(lambda * 0.5, 1e-12);
                    stepped = true;
                    break;
                }
                lambda *= 2.0;
            }
            if (!stepped) break;
            residualJac(free, goal, r, J); // refresh residual + Jacobian at new point
        }
        return free;
    };

    // Homotopy: morph the target from T_d (solved exactly by Phi = 0) to f,
    // warm-starting each step. This keeps every sub-problem near a known
    // solution, so it reaches high degree and targets near |f| = 1 where a
    // cold-start solve stalls.
    //
    // Phi = 0 is a stationary point of the (symmetric) response -- the exact
    // gradient there is zero -- so start just off it to give the solver a
    // descent direction.
    Eigen::VectorXd free = Eigen::VectorXd::Constant(nFree, 1e-2);
    const int steps = std::max(8, 2 * d);
    std::vector<double> goal(m);
    for (int st = 1; st <= steps; ++st) {
        const double s = static_cast<double>(st) / steps;
        for (int i = 0; i < m; ++i) goal[i] = (1.0 - s) * base[i] + s * fx[i];
        free = lmFit(goal, free);
    }

    QspSolveResult result;
    result.phases = expandSymmetric(free, d);
    result.iterations = steps;

    // Report the worst-case error over a fine grid on [-1, 1].
    double worst = 0.0;
    const int grid = 401;
    for (int i = 0; i < grid; ++i) {
        const double x = -1.0 + 2.0 * i / (grid - 1);
        worst = std::max(worst, std::abs(response(x, result.phases) - target(x)));
    }
    result.residual = worst;
    result.converged = worst < 1e-6;
    return result;
}

} // namespace qsvt
