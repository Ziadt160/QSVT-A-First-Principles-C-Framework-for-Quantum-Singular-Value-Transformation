// Standalone C++ smoke test for the extracted angle solver (no Python needed).
//
//   g++ -O3 -std=c++17 -I/usr/include/eigen3 -I../src
//       smoke.cpp ../src/QspAngleSolver.cpp -o smoke && ./smoke
//
// Verifies the vendored solver builds against Eigen via <Eigen/Dense> and
// reproduces a couple of known targets to machine precision.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "QspAngleSolver.hpp"

using qsvt::QspAngleSolver;

static int check(const char* name, double residual, double tol)
{
    const bool ok = residual < tol;
    std::printf("  %-24s residual=%.3e  %s\n", name, residual, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// Chebyshev T_d(x) via the trig identity (x clipped to [-1, 1]).
static double chebyshevT(int d, double x)
{
    const double xc = std::max(-1.0, std::min(1.0, x));
    return std::cos(d * std::acos(xc));
}

int main()
{
    int fails = 0;

    // FOUNDATIONAL INVARIANT: with all-zero phases the QSP response must equal
    // T_d(x) exactly. The homotopy starts from Phi = 0 == T_d and morphs to the
    // target, so if this drifts the whole solver's warm-start premise is wrong.
    {
        double worst = 0.0;
        for (int d : {1, 2, 3, 5, 8}) {
            const std::vector<double> zeros(d + 1, 0.0);
            for (int i = 0; i <= 40; ++i) {
                const double x = -1.0 + 2.0 * i / 40.0;
                const double got = QspAngleSolver::response(x, zeros);
                worst = std::max(worst, std::abs(got - chebyshevT(d, x)));
            }
        }
        fails += check("response(x,0)==T_d(x)", worst, 1e-12);
    }

    // Degree-1 odd target 0.7*x.
    {
        auto r = QspAngleSolver(1).solve([](double x) { return 0.7 * x; });
        double worst = 0.0;
        for (int i = 0; i <= 20; ++i) {
            const double x = -1.0 + 2.0 * i / 20.0;
            worst = std::max(worst, std::abs(QspAngleSolver::response(x, r.phases) - 0.7 * x));
        }
        fails += check("0.7*x (d=1)", worst, 1e-9);
    }

    // Degree-5 odd Chebyshev target 0.6*T_5(x).
    {
        auto f = [](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            return 0.6 * std::cos(5.0 * std::acos(xc));
        };
        auto r = QspAngleSolver(5).solve(f);
        fails += check("0.6*T_5 (d=5)", r.residual, 1e-6);
    }

    std::printf(fails == 0 ? "ALL C++ SMOKE TESTS PASSED\n" : "%d C++ SMOKE TEST(S) FAILED\n", fails);
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
