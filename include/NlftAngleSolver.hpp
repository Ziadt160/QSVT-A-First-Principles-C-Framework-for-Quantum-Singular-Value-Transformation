#pragma once
// End-to-end QSP angle finding by the inverse nonlinear Fourier transform --
// step 5 of the NLFT port (packages/qsp-angles/docs/NLFT_PORT_PLAN.md).
//
// Chain, mirroring nlft-qsp's chebqsp_solve:
//
//   Chebyshev coeffs  --to_laurent-->  Laurent P(z) = T((z+1/z)/2)
//                     --to_analytic--> analytic p(z) of degree n
//                     --(* -i)-->      QSP picture
//   Q = weissComplete(p)                 complementary, |Q|^2+|p|^2 = 1
//   c = ratio p/Q                        (same Weiss pass)
//   F = inverseNlft(p, c)                Half-Cholesky
//   phi_k = atan(Im F_k);  phi_last += pi/2      (from_nlfs, then iX)
//
// The result is in nlft-qsp's Chebyshev-QSP convention, which is NOT this
// project's Wx / Re<0|U|0> convention -- see qspPhasesFromNlft for that.
//
// COST: the Weiss step is O(1/eta) with eta = 1 - sup|P|, so this path is the
// wrong choice for targets pressed against |p| = 1. The symmetric-QSP Newton
// solver is eta-independent and remains the default; see NLFT_PORT_PLAN.md.

#include <vector>

namespace qsvt {

/// Phase factors in nlft-qsp's Chebyshev-QSP convention, for a real
/// definite-parity target given by its Chebyshev coefficients (low to high).
/// Reproduces `nlft_qsp.chebqsp_solve(coeffs).phi`.
std::vector<double> nlftChebQspPhases(const std::vector<double>& chebCoeffs);

} // namespace qsvt
