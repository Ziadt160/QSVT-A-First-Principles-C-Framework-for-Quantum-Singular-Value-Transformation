// Standalone C++ smoke test for the extracted angle solvers (no Python needed).
//
// Default build -- stdlib only, no third-party headers, no -I beyond ../src:
//
//   g++ -O3 -std=c++17 -I../src smoke.cpp ../src/SymQspAngleSolver.cpp
//       -o smoke && ./smoke
//
// With the optional Eigen-dependent homotopy fallback:
//
//   g++ -O3 -std=c++17 -DQSP_ANGLES_WITH_HOMOTOPY -I/usr/include/eigen3
//       -I../src smoke.cpp ../src/QspAngleSolver.cpp
//       ../src/SymQspAngleSolver.cpp -o smoke && ./smoke
//
// Verifies the vendored solvers reproduce known targets to machine precision.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "SymQspAngleSolver.hpp"
#ifdef QSP_ANGLES_WITH_HOMOTOPY
#include "QspAngleSolver.hpp"
using qsvt::QspAngleSolver;
#endif

using qsvt::SymQspAngleSolver;

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
    // (Checked through SymQspAngleSolver::response -- the two response
    // implementations are the same Wx product, so this holds for both.)
    {
        double worst = 0.0;
        for (int d : {1, 2, 3, 5, 8}) {
            const std::vector<double> zeros(d + 1, 0.0);
            for (int i = 0; i <= 40; ++i) {
                const double x = -1.0 + 2.0 * i / 40.0;
                const double got = SymQspAngleSolver::response(x, zeros);
                worst = std::max(worst, std::abs(got - chebyshevT(d, x)));
            }
        }
        fails += check("response(x,0)==T_d(x)", worst, 1e-12);
    }

#ifdef QSP_ANGLES_WITH_HOMOTOPY
    // --- homotopy fallback (optional; needs Eigen) ---------------------------

    // Both response implementations must agree bit-for-bit on the same phases --
    // this is what licenses using either one interchangeably above.
    {
        double worst = 0.0;
        for (const std::vector<double>& phi :
             {std::vector<double>{0.3, -0.2},
              std::vector<double>{0.1, 0.4, -0.7, 0.25},
              std::vector<double>{-0.5, 0.0, 0.9, -0.15, 0.6, 0.05}}) {
            for (int i = 0; i <= 40; ++i) {
                const double x = -1.0 + 2.0 * i / 40.0;
                worst = std::max(worst, std::abs(QspAngleSolver::response(x, phi)
                                                 - SymQspAngleSolver::response(x, phi)));
            }
        }
        fails += check("response impls agree", worst, 1e-15);
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
#endif // QSP_ANGLES_WITH_HOMOTOPY

    // --- sym_qsp (symmetric-QSP Newton method) ------------------------------
    // The phases it returns are in the SAME Re convention as QspAngleSolver, so
    // its self-reported residual already uses QspAngleSolver::response. These
    // hit machine precision (Newton, not homotopy), so the bar is much tighter.

    // Degree-5 ODD target 0.8*T_5(x).
    {
        auto f = [](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            return 0.8 * std::cos(5.0 * std::acos(xc));
        };
        auto r = SymQspAngleSolver(5).solve(f);
        fails += check("sym_qsp 0.8*T_5 (d=5)", r.residual, 1e-12);
    }

    // Degree-8 EVEN target 0.6*T_8(x).
    {
        auto f = [](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            return 0.6 * std::cos(8.0 * std::acos(xc));
        };
        auto r = SymQspAngleSolver(8).solve(f);
        fails += check("sym_qsp 0.6*T_8 (d=8)", r.residual, 1e-12);
    }

    // High degree: proves the Newton solver's degree reach (the homotopy solver
    // tops out near 100). Degree 301 still lands at machine precision.
    {
        const int d = 301;
        auto f = [d](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            return 0.8 * std::cos(d * std::acos(xc));
        };
        auto r = SymQspAngleSolver(d).solve(f);
        fails += check("sym_qsp 0.8*T_301 (d=301)", r.residual, 1e-10);
    }

    // GENERIC multi-coefficient targets -- the real QSVT case (many nonzero
    // Chebyshev modes, not a single T_d). Cross-validated against pyqsp on
    // random targets of both parities up to degree 101 (both ~machine precision).
    {
        // Odd, degree 9: 0.3 T1 - 0.2 T3 + 0.15 T5 - 0.1 T7 + 0.1 T9 (|f| < 1).
        const double c[5] = {0.3, -0.2, 0.15, -0.1, 0.1};
        auto f = [&](double x) {
            const double t = std::acos(std::max(-1.0, std::min(1.0, x)));
            double s = 0.0;
            for (int i = 0; i < 5; ++i) s += c[i] * std::cos((1 + 2 * i) * t);
            return s;
        };
        auto r = SymQspAngleSolver(9).solve(f);
        fails += check("sym_qsp generic odd (d=9)", r.residual, 1e-12);
    }
    {
        // Even, degree 8: 0.2 T0 + 0.25 T2 - 0.15 T4 + 0.1 T6 - 0.05 T8 (|f| < 1).
        const double c[5] = {0.2, 0.25, -0.15, 0.1, -0.05};
        auto f = [&](double x) {
            const double t = std::acos(std::max(-1.0, std::min(1.0, x)));
            double s = 0.0;
            for (int i = 0; i < 5; ++i) s += c[i] * std::cos((2 * i) * t);
            return s;
        };
        auto r = SymQspAngleSolver(8).solve(f);
        fails += check("sym_qsp generic even (d=8)", r.residual, 1e-12);
    }

    std::printf(fails == 0 ? "ALL C++ SMOKE TESTS PASSED\n" : "%d C++ SMOKE TEST(S) FAILED\n", fails);
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
