// C ABI implementation for qsp-angles. Wraps the validated C++
// SymQspAngleSolver; converts all C++ exceptions to status codes so nothing
// throws across the `extern "C"` boundary.

#include "qsp_angles.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "SymQspAngleSolver.hpp" // solve() and response() (Wx Re convention)

using qsvt::QspSolveResult;
using qsvt::SymQspAngleSolver;

namespace {

// Shared back end: solve a callable target and marshal results into C buffers.
qsp_status solve_impl(int degree, const std::function<double(double)>& f,
                      double* out_phases, int out_cap, int* out_len,
                      double* out_residual, int* out_converged)
{
    if (degree < 1) return QSP_ERR_DEGREE;
    if (out_phases == nullptr) return QSP_ERR_NULL;
    if (out_cap < degree + 1) return QSP_ERR_BUFFER;
    try {
        const QspSolveResult r = SymQspAngleSolver(degree).solve(f);
        const int n = static_cast<int>(r.phases.size());
        if (n > out_cap) return QSP_ERR_BUFFER;
        for (int i = 0; i < n; ++i) out_phases[i] = r.phases[i];
        if (out_len) *out_len = n;
        if (out_residual) *out_residual = r.residual;
        if (out_converged) *out_converged = r.converged ? 1 : 0;
        return QSP_OK;
    } catch (...) {
        return QSP_ERR_INTERNAL;
    }
}

} // namespace

extern "C" {

const char* qsp_version(void) { return "0.1.0"; }

const char* qsp_status_string(qsp_status status)
{
    switch (status) {
        case QSP_OK:               return "QSP_OK";
        case QSP_ERR_NULL:         return "QSP_ERR_NULL";
        case QSP_ERR_DEGREE:       return "QSP_ERR_DEGREE";
        case QSP_ERR_NCOEFFS:      return "QSP_ERR_NCOEFFS";
        case QSP_ERR_BUFFER:       return "QSP_ERR_BUFFER";
        case QSP_ERR_NOT_CONVERGED:return "QSP_ERR_NOT_CONVERGED";
        case QSP_ERR_INTERNAL:     return "QSP_ERR_INTERNAL";
    }
    return "QSP_ERR_UNKNOWN";
}

int qsp_num_phases(int degree) { return degree >= 1 ? degree + 1 : 0; }

qsp_status qsp_solve_chebyshev(int degree,
                               const double* cheb_coeffs, int ncoeffs,
                               double* out_phases, int out_cap, int* out_len,
                               double* out_residual, int* out_converged)
{
    if (cheb_coeffs == nullptr) return QSP_ERR_NULL;
    if (degree < 1) return QSP_ERR_DEGREE;
    if (ncoeffs != degree + 1) return QSP_ERR_NCOEFFS;

    // Copy the coefficients so the target callable owns its data.
    std::vector<double> c(cheb_coeffs, cheb_coeffs + ncoeffs);
    // Evaluate the Chebyshev series by Clenshaw recurrence: multiply-adds
    // instead of one std::cos per coefficient (the previous acos/cos form cost
    // O(d) transcendentals per evaluation, and the transform evaluates the
    // target at O(d) nodes).
    auto f = [c](double x) {
        const double xc = std::max(-1.0, std::min(1.0, x));
        double b1 = 0.0, b2 = 0.0;
        for (std::size_t i = c.size(); i-- > 1;) {
            const double b = 2.0 * xc * b1 - b2 + c[i];
            b2 = b1;
            b1 = b;
        }
        return c[0] + xc * b1 - b2;
    };
    return solve_impl(degree, f, out_phases, out_cap, out_len, out_residual, out_converged);
}

qsp_status qsp_solve_callback(int degree, qsp_target_fn f, void* ctx,
                              double* out_phases, int out_cap, int* out_len,
                              double* out_residual, int* out_converged)
{
    if (f == nullptr) return QSP_ERR_NULL;
    auto wrapped = [f, ctx](double x) { return f(x, ctx); };
    return solve_impl(degree, wrapped, out_phases, out_cap, out_len, out_residual, out_converged);
}

double qsp_response(double x, const double* phases, int len)
{
    if (phases == nullptr || len < 1) return 0.0;
    return SymQspAngleSolver::response(x, std::vector<double>(phases, phases + len));
}

} // extern "C"
