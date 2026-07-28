#include "InverseApproximation.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace qsvt {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Chebyshev-clustered abscissae: cos(pi j / (m-1)), j = 0..m-1, on [-1, 1].
std::vector<double> chebGrid(int m)
{
    std::vector<double> x(m);
    for (int j = 0; j < m; ++j) x[j] = std::cos(kPi * j / (m - 1));
    return x;
}

double clenshaw(const std::vector<double>& c, double x)
{
    const double xc = std::max(-1.0, std::min(1.0, x));
    double b1 = 0.0, b2 = 0.0;
    for (std::size_t i = c.size(); i-- > 1;) {
        const double b = 2.0 * xc * b1 - b2 + c[i];
        b2 = b1;
        b1 = b;
    }
    return c[0] + xc * b1 - b2;
}

} // namespace

double InverseApprox::operator()(double x) const { return clenshaw(chebCoeffs, x); }

InverseApprox approximateInverse(double kappa, int degree)
{
    if (degree % 2 == 0) --degree; // p must be odd
    if (degree < 1) degree = 1;

    const double delta = 1.0 / kappa;
    const double c0 = 1.0 / (2.0 * kappa); // any scale works; rescaled below
    const int nOdd = (degree + 1) / 2;     // T_1, T_3, ..., T_degree

    // Fit nodes: Chebyshev-clustered, mapped onto [delta, 1], oversampled ~4x so
    // the Vandermonde stays well conditioned at high degree.
    const int m = std::max(2000, 4 * degree + 2);
    const std::vector<double> u = chebGrid(m);
    Eigen::MatrixXd V(m, nOdd);
    Eigen::VectorXd y(m);
    for (int j = 0; j < m; ++j) {
        const double x = delta + (1.0 - delta) * (u[j] + 1.0) / 2.0;
        const double t = std::acos(std::max(-1.0, std::min(1.0, x)));
        for (int i = 0; i < nOdd; ++i) V(j, i) = std::cos((2 * i + 1) * t);
        y(j) = c0 / x;
    }
    const Eigen::VectorXd a = V.colPivHouseholderQr().solve(y);

    InverseApprox r;
    r.chebCoeffs.assign(degree + 1, 0.0);
    for (int i = 0; i < nOdd; ++i) r.chebCoeffs[2 * i + 1] = a(i);

    // Normalize by the TRUE sup on [-1, 1], measured on an endpoint-clustered
    // grid fine enough to resolve a degree-d polynomial's endpoint oscillation.
    const std::vector<double> xf = chebGrid(8 * degree + 2);
    double sup = 0.0;
    for (double x : xf) sup = std::max(sup, std::abs(clenshaw(r.chebCoeffs, x)));
    if (sup <= 0.0) sup = 1.0;
    for (double& v : r.chebCoeffs) v /= sup;
    r.c = c0 / sup;

    // Worst-case relative error against c/x on the domain.
    double rel = 0.0;
    for (double x : chebGrid(4 * degree + 2)) {
        if (x < delta) continue;
        const double want = r.c / x;
        rel = std::max(rel, std::abs(clenshaw(r.chebCoeffs, x) - want) / std::abs(want));
    }
    r.relErr = rel;
    return r;
}

double chebyshevSafetyScale(const std::function<double(double)>& f, int degree,
                            double safety)
{
    // Degree-`degree` Chebyshev truncation of f, on Chebyshev nodes.
    const int N = degree + 1;
    std::vector<double> c(N, 0.0), th(N), fv(N);
    for (int j = 0; j < N; ++j) {
        th[j] = kPi * (j + 0.5) / N;
        fv[j] = f(std::cos(th[j]));
    }
    for (int k = 0; k < N; ++k) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += fv[j] * std::cos(k * th[j]);
        c[k] = (k == 0 ? 1.0 / N : 2.0 / N) * s;
    }
    double sup = 0.0;
    for (double x : chebGrid(8 * degree + 2)) sup = std::max(sup, std::abs(clenshaw(c, x)));
    return sup > 1.0 ? safety / sup : safety;
}

int minInverseDegree(double kappa, double eps, int dmax)
{
    auto relAt = [&](int d) { return approximateInverse(kappa, d).relErr; };

    int lo = 1, hi = 3;
    if (relAt(lo) <= eps) return lo;
    while (hi <= dmax && relAt(hi) > eps) {
        lo = hi;
        hi *= 2;
    }
    if (hi > dmax && relAt(dmax) > eps) return -1;
    hi = std::min(hi, dmax);
    while (hi - lo > 2) {
        const int mid = ((lo + hi) / 2) | 1; // force odd
        if (relAt(mid) <= eps) hi = mid;
        else lo = mid;
    }
    return hi;
}

} // namespace qsvt
