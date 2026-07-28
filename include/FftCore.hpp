#pragma once
// Header-only, dependency-free complex FFT for the NLFT solver.
//
// Kept separate from SymQspAngleSolver.cpp's private copy on purpose: that file
// is vendored into packages/qsp-angles as a self-contained drop-in and is
// already validated, so it is not disturbed here. Deduplicating the two is a
// follow-up once the NLFT path is proven.
//
// Only power-of-two lengths are supported, which is all the Weiss/NLFT path
// needs (it always sizes transforms to a power of two).
//
// Sign convention, chosen to make the normalization explicit at every call
// site rather than baked in:
//   inverse = false ->  X[k] = sum_j a[j] exp(-2*pi*i*j*k/N)
//   inverse = true  ->  X[k] = sum_j a[j] exp(+2*pi*i*j*k/N)
// Neither direction divides by N.

#include <complex>
#include <cstddef>
#include <vector>

namespace qsvt {
namespace fftcore {

using cd = std::complex<double>;

inline bool isPowerOfTwo(std::size_t n) { return n && ((n & (n - 1)) == 0); }

inline std::size_t nextPowerOfTwo(std::size_t n)
{
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/// In-place iterative radix-2 Cooley-Tukey. `a.size()` must be a power of two.
inline void fftRadix2(std::vector<cd>& a, bool inverse)
{
    const std::size_t n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    constexpr double kPi = 3.14159265358979323846;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const cd wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const cd u = a[i + k];
                const cd v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

} // namespace fftcore
} // namespace qsvt
