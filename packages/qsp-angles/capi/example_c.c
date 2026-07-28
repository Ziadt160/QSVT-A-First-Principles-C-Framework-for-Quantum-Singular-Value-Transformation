/* example_c.c - a PURE C program using the qsp-angles C ABI.
 *
 * This file is compiled by a C compiler (gcc/clang) and knows nothing about
 * C++ or Python. It links the qsp-angles C API and solves for the QSP phases of
 * a target polynomial -- the proof that the solver is embeddable.
 *
 * Nothing here needs a third-party library: the solver behind the ABI is
 * stdlib-only C++17, so the C++ half compiles with no -I beyond ../src.
 *
 * Build (see capi/README.md):
 *   g++ -O3 -std=c++17 -I../src -c qsp_angles.cpp ../src/SymQspAngleSolver.cpp
 *   gcc -O2 example_c.c qsp_angles.o SymQspAngleSolver.o \
 *       -lstdc++ -lm -o example_c && ./example_c
 */
#include <math.h>
#include <stdio.h>

#include "qsp_angles.h"

int main(void)
{
    /* Target: 0.8 * T_5(x). Full Chebyshev coefficients c_0..c_5. */
    const int degree = 5;
    const double coeffs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.8};

    double phases[16];
    int len = 0, converged = 0;
    double residual = 0.0;

    qsp_status st = qsp_solve_chebyshev(degree, coeffs, 6,
                                        phases, 16, &len, &residual, &converged);
    if (st != QSP_OK) {
        printf("solve failed: %s\n", qsp_status_string(st));
        return 1;
    }

    printf("qsp-angles %s (pure-C client)\n", qsp_version());
    printf("degree %d -> %d phases, residual = %.2e, converged = %d\n",
           degree, len, residual, converged);
    printf("phases:");
    for (int i = 0; i < len; ++i) printf(" %.5f", phases[i]);
    printf("\n");

    /* Verify the phases reproduce 0.8*T_5(x) via Re<0|U(x)|0>. */
    double max_err = 0.0;
    for (int i = 0; i <= 20; ++i) {
        double x = -1.0 + 2.0 * i / 20.0;
        double xc = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
        double got = qsp_response(x, phases, len);
        double want = 0.8 * cos(5.0 * acos(xc));
        double e = fabs(got - want);
        if (e > max_err) max_err = e;
    }
    printf("max |Re<0|U|0> - 0.8*T_5| over grid = %.2e\n", max_err);

    return max_err < 1e-10 ? 0 : 2;
}
