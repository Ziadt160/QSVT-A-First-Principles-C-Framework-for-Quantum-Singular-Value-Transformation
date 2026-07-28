// Isolate the cause: at FIXED degree, does our Newton cost grow with target
// difficulty (kappa)? Reports iteration count and wall-clock per kappa.
#include <chrono>
#include <cstdio>

#include "InverseApproximation.hpp"
#include "SymQspAngleSolver.hpp"

int main()
{
    const int degree = 1001;
    std::printf("degree fixed at %d; varying target difficulty (kappa)\n\n", degree);
    std::printf("%8s %7s %12s %11s\n", "kappa", "iters", "solve_s", "residual");
    for (double kap : {10.0, 30.0, 60.0, 110.0, 165.0}) {
        const qsvt::InverseApprox inv = qsvt::approximateInverse(kap, degree);
        auto target = [&inv](double x) { return 0.999 * inv(x); };
        const auto t0 = std::chrono::steady_clock::now();
        const qsvt::QspSolveResult r =
            qsvt::SymQspAngleSolver(degree).solve(target);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        std::printf("%8.0f %7d %12.3f %11.2e\n", kap, r.iterations, secs,
                    r.residual);
        std::fflush(stdout);
    }
    return 0;
}
