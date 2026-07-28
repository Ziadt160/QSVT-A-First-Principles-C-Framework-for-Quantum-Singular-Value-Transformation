#include "NlftAngleSolver.hpp"

#include <cmath>
#include <stdexcept>

#include "InverseNlft.hpp"
#include "LaurentPoly.hpp"
#include "WeissCompletion.hpp"

namespace qsvt {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// T(x) with x = (z + 1/z)/2, as a Laurent polynomial.
/// Coefficient of z^k is c_{|k|}/2 for k != 0, and c_0 for k = 0.
LaurentPoly chebToLaurent(const std::vector<double>& c)
{
    const int n = static_cast<int>(c.size()) - 1;
    std::vector<Complexd> out(static_cast<std::size_t>(2 * n + 1), Complexd(0.0, 0.0));
    for (int k = -n; k <= n; ++k) {
        const double v = c[static_cast<std::size_t>(std::abs(k))];
        out[static_cast<std::size_t>(k + n)] = Complexd(k == 0 ? v : 0.5 * v, 0.0);
    }
    return LaurentPoly(std::move(out), -n);
}

/// Definite-parity Laurent p[0] z^-n + p[1] z^(-n+2) + ... -> analytic
/// p[0] + p[1] z + ... + p[n] z^n.
LaurentPoly laurentToAnalytic(const LaurentPoly& P)
{
    const int n = std::max(std::abs(P.supportStart), std::abs(P.supportEnd() - 1));
    std::vector<Complexd> out(static_cast<std::size_t>(n) + 1);
    for (int k = 0; k <= n; ++k) out[static_cast<std::size_t>(k)] = P.at(2 * k - n);
    return LaurentPoly(std::move(out), 0);
}

} // namespace

std::vector<double> nlftChebQspPhases(const std::vector<double>& chebCoeffs)
{
    if (chebCoeffs.empty()) return {};

    LaurentPoly p = laurentToAnalytic(chebToLaurent(chebCoeffs));

    // QSP picture: (Q, -iP) -> (P, iQ).
    for (Complexd& v : p.coeffs) v *= Complexd(0.0, -1.0);

    const WeissResult w = weissComplete(p, -1.0, /*withRatio=*/true);
    const NlftSequence F = inverseNlft(p, w.c);

    // from_nlfs: phi_k = arctan(Im F_k)   (F must be purely imaginary).
    std::vector<double> phi(F.coeffs.size());
    for (std::size_t k = 0; k < F.coeffs.size(); ++k) {
        if (std::abs(F.coeffs[k].real()) > 1e-6 * (1.0 + std::abs(F.coeffs[k].imag()))) {
            throw std::runtime_error(
                "nlftChebQspPhases: NLFT sequence is not purely imaginary");
        }
        phi[k] = std::atan(F.coeffs[k].imag());
    }

    // iX: multiply the protocol by iX on the right.
    if (!phi.empty()) phi.back() += kPi / 2.0;

    return phi;
}

} // namespace qsvt
