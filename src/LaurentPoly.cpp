#include "LaurentPoly.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "FftCore.hpp"

namespace qsvt {

namespace {
constexpr double kPi = 3.14159265358979323846;
} // namespace

int LaurentPoly::effectiveDegree(double tol) const
{
    int deg = 0;
    for (int k = supportStart; k < supportEnd(); ++k) {
        if (std::abs(at(k)) > tol) deg = std::max(deg, std::abs(k));
    }
    return deg;
}

std::vector<Complexd> LaurentPoly::evalAtRootsOfUnity(std::size_t N) const
{
    if (!fftcore::isPowerOfTwo(N)) {
        throw std::invalid_argument("evalAtRootsOfUnity: N must be a power of two");
    }
    if (coeffs.size() > N) {
        throw std::invalid_argument("evalAtRootsOfUnity: N smaller than the support");
    }

    // p(w^k) = w^(k*supportStart) * sum_j coeffs[j] w^(k*j), w = exp(2*pi*i/N).
    std::vector<Complexd> buf(N, Complexd(0.0, 0.0));
    std::copy(coeffs.begin(), coeffs.end(), buf.begin());
    fftcore::fftRadix2(buf, /*inverse=*/true); // sum_j c_j exp(+2 pi i j k / N)

    for (std::size_t k = 0; k < N; ++k) {
        const double ang = 2.0 * kPi * static_cast<double>(k) *
                           static_cast<double>(supportStart) / static_cast<double>(N);
        buf[k] *= Complexd(std::cos(ang), std::sin(ang));
    }
    return buf;
}

double LaurentPoly::supNorm(std::size_t N) const
{
    const std::vector<Complexd> pts = evalAtRootsOfUnity(N);
    double m = 0.0;
    for (const Complexd& v : pts) m = std::max(m, std::abs(v));
    return m;
}

double LaurentPoly::l2Norm() const
{
    double s = 0.0;
    for (const Complexd& c : coeffs) s += std::norm(c);
    return std::sqrt(s);
}

LaurentPoly LaurentPoly::conjugate() const
{
    // p*(z) = sum_j conj(c_j) z^(-j): reverse and conjugate.
    std::vector<Complexd> out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        out[coeffs.size() - 1 - i] = std::conj(coeffs[i]);
    }
    return LaurentPoly(std::move(out), -(supportEnd() - 1));
}

LaurentPoly LaurentPoly::truncate(int lo, int hi) const
{
    if (hi < lo) return LaurentPoly({}, 0);
    std::vector<Complexd> out(static_cast<std::size_t>(hi - lo + 1), Complexd(0.0, 0.0));
    for (int k = lo; k <= hi; ++k) out[static_cast<std::size_t>(k - lo)] = at(k);
    return LaurentPoly(std::move(out), lo);
}

LaurentPoly LaurentPoly::operator*(const LaurentPoly& other) const
{
    if (coeffs.empty() || other.coeffs.empty()) return LaurentPoly({}, 0);
    std::vector<Complexd> out(coeffs.size() + other.coeffs.size() - 1,
                              Complexd(0.0, 0.0));
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        for (std::size_t j = 0; j < other.coeffs.size(); ++j) {
            out[i + j] += coeffs[i] * other.coeffs[j];
        }
    }
    return LaurentPoly(std::move(out), supportStart + other.supportStart);
}

LaurentPoly LaurentPoly::operator+(const LaurentPoly& other) const
{
    if (coeffs.empty()) return other;
    if (other.coeffs.empty()) return *this;
    const int lo = std::min(supportStart, other.supportStart);
    const int hi = std::max(supportEnd(), other.supportEnd()) - 1;
    std::vector<Complexd> out(static_cast<std::size_t>(hi - lo + 1), Complexd(0.0, 0.0));
    for (int k = lo; k <= hi; ++k) {
        out[static_cast<std::size_t>(k - lo)] = at(k) + other.at(k);
    }
    return LaurentPoly(std::move(out), lo);
}

LaurentPoly LaurentPoly::minusScalar(Complexd s) const
{
    LaurentPoly r = *this;
    if (0 >= r.supportStart && 0 < r.supportEnd()) {
        r.coeffs[static_cast<std::size_t>(-r.supportStart)] -= s;
        return r;
    }
    return r + LaurentPoly({-s}, 0);
}

LaurentPoly laurentApproximation(const std::vector<Complexd>& points)
{
    const std::size_t N = points.size();
    if (!fftcore::isPowerOfTwo(N)) {
        throw std::invalid_argument("laurentApproximation: N must be a power of two");
    }

    // Forward-normalized DFT: c[m] = (1/N) sum_k f(w^k) exp(-2 pi i k m / N).
    std::vector<Complexd> c = points;
    fftcore::fftRadix2(c, /*inverse=*/false);
    const double inv = 1.0 / static_cast<double>(N);
    for (Complexd& v : c) v *= inv;

    // Shift zero frequency to the middle: shifted[j] = c[(j + N/2) mod N], so the
    // stored coefficients run over exponents [-N/2, N/2).
    const std::size_t half = N / 2;
    std::vector<Complexd> shifted(N);
    for (std::size_t j = 0; j < N; ++j) shifted[j] = c[(j + half) % N];

    return LaurentPoly(std::move(shifted), -static_cast<int>(half));
}

LaurentPoly schwarzTransform(const LaurentPoly& p)
{
    // Keep exponents k <= 0: double the k < 0 terms, leave k = 0 as is, drop k > 0.
    const int lo = p.supportStart;
    const int hi = std::min(0, p.supportEnd() - 1);
    if (hi < lo) return LaurentPoly({}, 0);

    std::vector<Complexd> out(static_cast<std::size_t>(hi - lo + 1), Complexd(0.0, 0.0));
    for (int k = lo; k <= hi; ++k) {
        out[static_cast<std::size_t>(k - lo)] = (k < 0) ? 2.0 * p.at(k) : p.at(k);
    }
    return LaurentPoly(std::move(out), lo);
}

} // namespace qsvt
