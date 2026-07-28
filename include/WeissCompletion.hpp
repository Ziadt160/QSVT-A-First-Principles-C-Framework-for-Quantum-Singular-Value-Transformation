#pragma once
// Weiss' algorithm: given b, find the complementary polynomial a with
// |a|^2 + |b|^2 = 1 on the unit circle -- step 3 of the NLFT port
// (packages/qsp-angles/docs/NLFT_PORT_PLAN.md).
//
// a is the unique OUTER, positive-mean solution (arXiv:2407.05634). It is
// obtained from the outer function of sqrt(1 - |b|^2):
//
//     R = log(1 - |b|^2) / 2      on the circle
//     G = Schwarz transform of R  (analytic completion; Re G = R)
//     a = exp(G),                 truncated back to b's support
//
// evaluated by FFT at N roots of unity, with N doubled until the residual
// || a a* + b b* - 1 ||_2 stops improving.
//
// COST NOTE, and it drives the whole dispatch strategy: the starting N is
// ~ d/eta with eta = 1 - sup|b|. So this method is O(1/eta) -- targets pressed
// against |b| = 1 are expensive here, while the symmetric-QSP Newton solver is
// eta-independent. See NLFT_PORT_PLAN.md.

#include <stdexcept>

#include "LaurentPoly.hpp"

namespace qsvt {

class WeissConvergenceError : public std::runtime_error {
public:
    explicit WeissConvergenceError(const std::string& what)
        : std::runtime_error(what)
    {
    }
};

struct WeissResult {
    LaurentPoly a;      ///< complementary polynomial, support [-deg(b), 0]
    LaurentPoly c;      ///< approximation of b/a (only if requested)
    double residual{0}; ///< || a a* + b b* - 1 ||_2
    std::size_t fftSize{0};
    int rounds{0};
};

/// Weiss completion of @p b.
/// @param eps      target residual; <= 0 selects 100 * machine epsilon.
/// @param withRatio also compute c ~ b/a (needed by the inverse NLFT).
/// @throws WeissConvergenceError if the residual stops improving.
WeissResult weissComplete(const LaurentPoly& b, double eps = -1.0,
                          bool withRatio = false);

} // namespace qsvt
