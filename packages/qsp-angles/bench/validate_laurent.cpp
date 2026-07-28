// Step-1 validation for the NLFT port: LaurentPoly vs nlft-qsp's Polynomial.
//
// Emits its results as machine-readable lines so validate_laurent.py can compare
// them against the Python reference. Run via that script, not directly.
#include <cstdio>
#include <random>
#include <vector>

#include "LaurentPoly.hpp"

using qsvt::Complexd;
using qsvt::LaurentPoly;

static void dump(const char* tag, const LaurentPoly& p)
{
    std::printf("%s %d %zu", tag, p.supportStart, p.coeffs.size());
    for (const Complexd& c : p.coeffs) {
        std::printf(" %.17g %.17g", c.real(), c.imag());
    }
    std::printf("\n");
}

int main()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    // A deterministic pseudo-random Laurent polynomial with support [-4, 3].
    const int start = -4;
    std::vector<Complexd> c;
    for (int i = 0; i < 8; ++i) c.emplace_back(uni(rng), uni(rng));
    const LaurentPoly p(c, start);
    dump("POLY", p);

    // Values on 16 roots of unity.
    const std::vector<Complexd> pts = p.evalAtRootsOfUnity(16);
    std::printf("EVAL16 %zu", pts.size());
    for (const Complexd& v : pts) std::printf(" %.17g %.17g", v.real(), v.imag());
    std::printf("\n");

    // Round-trip: interpolating those values must return the same polynomial
    // (up to the frequency shift), which pins down both conventions at once.
    dump("APPROX", qsvt::laurentApproximation(pts));

    dump("SCHWARZ", qsvt::schwarzTransform(p));
    dump("CONJ", p.conjugate());
    dump("PROD", p * p.conjugate());
    std::printf("SUPNORM %.17g\n", p.supNorm(1024));
    std::printf("L2 %.17g\n", p.l2Norm());
    return 0;
}
