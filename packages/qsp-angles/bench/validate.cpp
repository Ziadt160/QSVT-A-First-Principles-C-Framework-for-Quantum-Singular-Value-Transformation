// Independent validation driver: solve a manifest of generic multi-coefficient
// targets with the C++ sym_qsp solver and print the achieved residual per case.
//
// The manifest (argv[1]) is produced by validate_vs_pyqsp.py and contains, for
// each case, a line "parity degree n" followed by n reduced Chebyshev
// coefficients (the nonzero, parity-selected modes, low->high). The same
// coefficients are solved by pyqsp on the Python side; comparing the two
// residuals is a convention-neutral, third-party-reproducible correctness check
// (each solver graded in its own convention against the same target).
//
//   g++ -O3 -march=native -std=c++17 -I../src -I/usr/include/eigen3
//       validate.cpp ../src/SymQspAngleSolver.cpp ../src/QspAngleSolver.cpp -o validate
//   ./validate cases.txt

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "SymQspAngleSolver.hpp"

using qsvt::SymQspAngleSolver;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <manifest>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open manifest: %s\n", argv[1]);
        return 2;
    }

    std::printf("degree,parity,cpp_residual,converged\n");
    int parity, degree, n;
    while (in >> parity >> degree >> n) {
        std::vector<double> coef(n);
        for (int i = 0; i < n; ++i) in >> coef[i];

        // Target as an explicit parity-reduced Chebyshev series.
        auto f = [coef, parity](double x) {
            const double xc = std::max(-1.0, std::min(1.0, x));
            const double t = std::acos(xc);
            double s = 0.0;
            for (std::size_t i = 0; i < coef.size(); ++i)
                s += coef[i] * std::cos((parity + 2 * static_cast<int>(i)) * t);
            return s;
        };
        const auto r = SymQspAngleSolver(degree).solve(f);
        std::printf("%d,%d,%.3e,%d\n", degree, parity, r.residual, r.converged ? 1 : 0);
    }
    return 0;
}
