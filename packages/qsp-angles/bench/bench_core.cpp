// Timing/accuracy harness for the C++ core, across degrees, for a scaled
// Chebyshev target c*T_d (parity d%2, |f|<=1). Isolates angle-FINDING from
// polynomial approximation. Prints CSV: degree,core_time_ms,core_residual.
//
//   g++ -O3 -march=native -DNDEBUG -std=c++17 -I../src -I/usr/include/eigen3
//       bench_core.cpp ../src/QspAngleSolver.cpp -o bench_core && ./bench_core
//
// Pair with bench_pyqsp.py and compare; see BENCHMARK.md.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "QspAngleSolver.hpp"

using qsvt::QspAngleSolver;
using clk = std::chrono::high_resolution_clock;

int main()
{
    const double C = 0.8;
    const int degs[] = {11, 21, 31, 51, 71, 101};

    std::printf("degree,core_time_ms,core_residual\n");
    for (int d : degs) {
        auto f = [&](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            return C * std::cos(d * std::acos(xc));
        };
        std::vector<double> ts;
        double res = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = clk::now();
            const auto r = QspAngleSolver(d).solve(f);
            const auto t1 = clk::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            res = r.residual;
        }
        std::sort(ts.begin(), ts.end());
        std::printf("%d,%.1f,%.2e\n", d, ts[1], res); // median of 3
    }
    return 0;
}
