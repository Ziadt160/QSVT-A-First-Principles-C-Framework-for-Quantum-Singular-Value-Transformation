// Timing/accuracy harness for the C++ sym_qsp core (the symmetric-QSP Newton
// method), across degrees, for a scaled Chebyshev target c*T_d (parity d%2,
// |f|<=1). Prints CSV: degree,iters,time_ms,residual.
//
//   g++ -O3 -march=native -DNDEBUG -std=c++17 -I../src -I/usr/include/eigen3
//       bench_symqsp_core.cpp ../src/SymQspAngleSolver.cpp
//       ../src/QspAngleSolver.cpp -o bench_symqsp_core && ./bench_symqsp_core
//
// This is the Newton solver, so it stays at ~4-5 iterations and machine
// precision across degrees, and reaches degree > 1000 -- see BENCHMARK.md for
// the head-to-head against pyqsp's sym_qsp.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "SymQspAngleSolver.hpp"

using qsvt::SymQspAngleSolver;
using clk = std::chrono::high_resolution_clock;

int main()
{
    const double C = 0.8;
    const int degs[] = {11, 21, 51, 101, 201, 501, 1001};

    std::printf("degree,iters,time_ms,residual\n");
    for (int d : degs) {
        auto f = [&](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            return C * std::cos(d * std::acos(xc));
        };
        std::vector<double> ts;
        double res = 0.0;
        int iters = 0;
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = clk::now();
            const auto r = SymQspAngleSolver(d).solve(f);
            const auto t1 = clk::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            res = r.residual;
            iters = r.iterations;
        }
        std::sort(ts.begin(), ts.end());
        std::printf("%d,%d,%.1f,%.2e\n", d, iters, ts[1], res); // median of 3
    }
    return 0;
}
