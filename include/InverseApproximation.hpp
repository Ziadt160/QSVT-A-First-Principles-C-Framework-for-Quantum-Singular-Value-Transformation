#pragma once
// Near-minimax odd polynomial approximation of c/x -- the standard QSVT
// matrix-inversion target.
//
// Why this exists: the naive choice of a smoothed rational target such as
// f(x) = c*x/(x^2 + eps) is NOT a polynomial, and a QSP solver is graded against
// it across the whole of [-1, 1] -- including the sharp feature at x = 0, which
// no moderate-degree polynomial can reproduce. That produces a large "fit
// residual" that looks like a solver failure but is really a target-construction
// failure.
//
// The correct construction approximates c/x only on the domain that matters,
// D = [1/kappa, 1] u [-1, -1/kappa], in the odd Chebyshev basis, then rescales
// so max|p| <= 1 on [-1, 1] (the QSP validity condition).
//
// Both the fit nodes and the sup-norm grid are CHEBYSHEV-clustered (x = cos t).
// This matters above degree ~600: a uniform grid makes the Vandermonde system
// ill-conditioned, the fit oscillates to O(10^2) *between* nodes near x = +-1,
// and a uniform sup grid steps straight over those spikes -- so the "normalised"
// coefficients silently describe a polynomial with true sup-norm >> 1, i.e. an
// invalid QSP target that no phase sequence can realise.

#include <functional>
#include <vector>

namespace qsvt {

/// An odd polynomial p with p(x) ~ c/x on [1/kappa, 1] u [-1, -1/kappa] and
/// |p(x)| <= 1 on [-1, 1].
struct InverseApprox {
    /// Full Chebyshev coefficients for T_0 .. T_degree (even entries are zero).
    std::vector<double> chebCoeffs;
    /// Subnormalization: p(x) ~ c / x on the domain. Sets the QSVT success
    /// amplitude (~c * ||A^{-1} b||), so larger is better.
    double c = 0.0;
    /// Worst-case RELATIVE error of p against c/x on the domain.
    double relErr = 0.0;

    /// Evaluate p(x) by Clenshaw recurrence.
    double operator()(double x) const;
};

/// Least-squares odd-Chebyshev fit of c/x on D, normalized to unit sup-norm.
/// @param kappa Condition number; the domain is [1/kappa, 1] and its reflection.
/// @param degree Polynomial degree (an even value is reduced by one: p is odd).
InverseApprox approximateInverse(double kappa, int degree);

/// Smallest odd degree whose relative error on D is <= eps, or -1 if not
/// reached by @p dmax. Uses a geometric bracket then bisection.
int minInverseDegree(double kappa, double eps, int dmax = 4001);

/// Scale s such that s * (degree-`degree` Chebyshev truncation of @p f) satisfies
/// |p| <= 1 on [-1, 1] -- the QSP validity condition.
///
/// Targets that approach +-1 asymptotically (erf and other sign approximations)
/// always overshoot slightly under truncation, so this shave is mandatory rather
/// than cosmetic: a symmetric-QSP Newton solve on an |p| > 1 target has no
/// solution and will diverge. Returns @p safety when no rescaling is needed.
double chebyshevSafetyScale(const std::function<double(double)>& f, int degree,
                            double safety = 0.999);

} // namespace qsvt
