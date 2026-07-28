#include "WeissCompletion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "FftCore.hpp"

namespace qsvt {

namespace {

constexpr int kMaxAttempts = 3; // matches nlft-qsp's WEISS_MAX_ATTEMPTS

/// || a a* + b b* - 1 ||_2, the defining Weiss residual.
double completionResidual(const LaurentPoly& a, const LaurentPoly& b)
{
    const LaurentPoly lhs = (a * a.conjugate()) + (b * b.conjugate());
    return lhs.minusScalar(Complexd(1.0, 0.0)).l2Norm();
}

} // namespace

WeissResult weissComplete(const LaurentPoly& b, double eps, bool withRatio)
{
    const int d = b.effectiveDegree();
    if (eps <= 0.0) eps = 100.0 * std::numeric_limits<double>::epsilon();

    // eta = 1 - sup|b|. The starting transform length scales as d/eta, so a
    // target that touches |b| = 1 makes this method very expensive.
    const std::size_t supSamples =
        fftcore::nextPowerOfTwo(static_cast<std::size_t>(std::max(4000, 4 * d)));
    const double sup = b.supNorm(supSamples);
    const double eta = 1.0 - sup;
    if (eta <= 0.0) {
        throw WeissConvergenceError(
            "weissComplete: sup|b| >= 1, no complementary polynomial exists");
    }

    std::size_t N = fftcore::nextPowerOfTwo(
                        static_cast<std::size_t>(static_cast<double>(d) / eta)) /
                    2;
    if (N < 2) N = 2;

    double threshold = 1.0;
    int attempts = 0;
    WeissResult best;
    best.residual = threshold;

    while (threshold > eps) {
        N *= 2;
        if (N > (std::size_t(1) << 26)) {
            throw WeissConvergenceError(
                "weissComplete: transform length exceeded 2^26 before converging");
        }

        const std::vector<Complexd> bPoints = b.evalAtRootsOfUnity(N);

        // R = log(1 - |b|^2) / 2 on the circle.
        std::vector<Complexd> rPoints(N);
        for (std::size_t k = 0; k < N; ++k) {
            const double m = 1.0 - std::norm(bPoints[k]);
            if (m <= 0.0) {
                throw WeissConvergenceError(
                    "weissComplete: 1 - |b|^2 is non-positive on the circle");
            }
            rPoints[k] = Complexd(0.5 * std::log(m), 0.0);
        }

        const LaurentPoly R = laurentApproximation(rPoints);
        const LaurentPoly G = schwarzTransform(R);
        const std::vector<Complexd> gPoints = G.evalAtRootsOfUnity(N);

        std::vector<Complexd> aPoints(N);
        for (std::size_t k = 0; k < N; ++k) aPoints[k] = std::exp(gPoints[k]);

        LaurentPoly a = laurentApproximation(aPoints).truncate(-d, 0);

        const double newThr = completionResidual(a, b);

        if (threshold <= newThr) {
            // Plateau. Unlike nlft-qsp -- which runs on an arbitrary-precision
            // (mpmath) backend by default and can drive this to ~1e-19 -- this
            // is a double-precision implementation, so the attainable residual
            // is bounded by conditioning and degrades with degree. Stopping at
            // the best value found is the correct behaviour here; only a result
            // that never became usable is an error.
            if (++attempts >= kMaxAttempts) {
                if (best.a.coeffs.empty()) {
                    throw WeissConvergenceError(
                        "weissComplete: no usable completion found");
                }
                break;
            }
        } else {
            threshold = newThr;
            attempts = 0;
            best.a = a;
            best.residual = newThr;
            best.fftSize = N;
            if (withRatio) {
                // c ~ b/a, computed as b * exp(-G) on the circle.
                std::vector<Complexd> cPoints(N);
                for (std::size_t k = 0; k < N; ++k) {
                    cPoints[k] = bPoints[k] * std::exp(-gPoints[k]);
                }
                const LaurentPoly craw = laurentApproximation(cPoints);
                best.c = craw.truncate(craw.supportStart, b.supportEnd() - 1);
            }
        }
        ++best.rounds;
    }

    return best;
}

} // namespace qsvt
