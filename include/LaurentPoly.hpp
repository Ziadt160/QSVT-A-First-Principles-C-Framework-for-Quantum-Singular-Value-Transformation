#pragma once
// Laurent polynomials on the unit circle -- step 1 of the NLFT port
// (see packages/qsp-angles/docs/NLFT_PORT_PLAN.md).
//
// A LaurentPoly represents  p(z) = sum_{j} coeffs[j] * z^(supportStart + j),
// i.e. `supportStart` is the exponent of the first stored coefficient and may be
// negative. This mirrors nlft-qsp's `Polynomial(coeffs, support_start)` so the
// two can be validated coefficient-wise against each other.
//
// Dependency-free: standard library plus the header-only FFT in FftCore.hpp.

#include <complex>
#include <cstddef>
#include <vector>

namespace qsvt {

using Complexd = std::complex<double>;

struct LaurentPoly {
    std::vector<Complexd> coeffs;
    int supportStart{0};

    LaurentPoly() = default;
    LaurentPoly(std::vector<Complexd> c, int start)
        : coeffs(std::move(c)), supportStart(start)
    {
    }

    /// One past the highest stored exponent.
    int supportEnd() const { return supportStart + static_cast<int>(coeffs.size()); }

    /// Coefficient of z^k (zero outside the stored support).
    Complexd at(int k) const
    {
        if (k < supportStart || k >= supportEnd()) return Complexd(0.0, 0.0);
        return coeffs[static_cast<std::size_t>(k - supportStart)];
    }

    /// Largest |exponent| carrying a non-negligible coefficient.
    int effectiveDegree(double tol = 1e-14) const;

    /// Values at the N-th roots of unity: result[k] = p(exp(2*pi*i*k/N)).
    /// N must be a power of two and at least the support width.
    std::vector<Complexd> evalAtRootsOfUnity(std::size_t N) const;

    /// max |p| sampled over N roots of unity.
    double supNorm(std::size_t N) const;

    /// sqrt(sum |coeff|^2) -- equals the L2 norm on the circle by Parseval.
    double l2Norm() const;

    /// p*(z) = conj(p(1/conj(z))); on |z| = 1 this is the pointwise conjugate.
    /// Coefficients are reversed and conjugated.
    LaurentPoly conjugate() const;

    /// Restrict to exponents [lo, hi] (inclusive), dropping the rest.
    LaurentPoly truncate(int lo, int hi) const;

    /// Polynomial product (coefficient convolution).
    LaurentPoly operator*(const LaurentPoly& other) const;

    /// Coefficient-wise sum.
    LaurentPoly operator+(const LaurentPoly& other) const;

    /// Subtract a scalar (i.e. from the z^0 coefficient).
    LaurentPoly minusScalar(Complexd s) const;
};

/// The unique Laurent polynomial through `points` at the N-th roots of unity,
/// with frequencies shifted into [-N/2, N/2). `points.size()` must be a power of
/// two. Mirrors nlft-qsp's `weiss.laurent_approximation`.
LaurentPoly laurentApproximation(const std::vector<Complexd>& points);

/// The anti-analytic polynomial whose real part is `p` on the circle: keeps the
/// exponents k <= 0, doubling those with k < 0 and leaving k = 0 alone. Adding
/// i*H[p] with H the Hilbert transform. Mirrors `Polynomial.schwarz_transform`.
LaurentPoly schwarzTransform(const LaurentPoly& p);

} // namespace qsvt
