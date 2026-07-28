// pybind11 module for qsp-angles: exposes only the standalone QSP angle solver.
//
// The heavy QSVT/Qrack stack is intentionally NOT here -- this package is the
// extracted angle-finding wedge.
//
// DEPENDENCIES: the default build is stdlib-only C++17 (plus pybind11). The
// homotopy-continuation fallback needs Eigen, so it is compiled in only when
// QSP_ANGLES_WITH_HOMOTOPY is defined (CMake: -DQSP_ANGLES_WITH_HOMOTOPY=ON).
// Without it there is nothing to find_package and nothing to fetch, so a wheel
// builds offline on a bare machine. `has_homotopy` reports which build this is.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "SymQspAngleSolver.hpp"
#ifdef QSP_ANGLES_WITH_HOMOTOPY
#include "QspAngleSolver.hpp"
#endif

namespace py = pybind11;
using namespace qsvt;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Fast C++ QSP angle solvers: a symmetric-QSP Newton method "
              "(sym_qsp, default) and an optional homotopy-continuation "
              "fallback (built only with QSP_ANGLES_WITH_HOMOTOPY).";

#ifdef QSP_ANGLES_WITH_HOMOTOPY
    m.attr("has_homotopy") = true;

    m.def(
        "poly_to_angles",
        [](const std::function<double(double)>& target, int degree) {
            const QspSolveResult r = QspAngleSolver(degree).solve(target);
            py::dict d;
            d["phases"] = r.phases;
            d["residual"] = r.residual;
            d["converged"] = r.converged;
            return d;
        },
        py::arg("target"), py::arg("degree"),
        "QSP phases realising a degree-d polynomial approximation of target(x) "
        "in the Wx convention, via the homotopy-continuation solver. Returns "
        "{phases, residual, converged}.");
#else
    m.attr("has_homotopy") = false;
#endif

    m.def(
        "sym_qsp_poly_to_angles",
        [](const std::function<double(double)>& target, int degree) {
            const QspSolveResult r = SymQspAngleSolver(degree).solve(target);
            py::dict d;
            d["phases"] = r.phases;
            d["residual"] = r.residual;
            d["converged"] = r.converged;
            return d;
        },
        py::arg("target"), py::arg("degree"),
        "QSP phases realising a degree-d polynomial approximation of target(x) "
        "in the Wx convention, via the symmetric-QSP Newton method (sym_qsp; "
        "Dong-Lin-Ni-Wang, arXiv:2307.12468). Faster and reaches higher degree "
        "than the homotopy fallback. Returns {phases, residual, converged}.");

    // Convention-identical to QspAngleSolver::response (same Wx product, same
    // Re<0|U|0>), but stdlib-only -- so `response` exists in every build.
    m.def(
        "response",
        [](double x, const std::vector<double>& phases) {
            return SymQspAngleSolver::response(x, phases);
        },
        py::arg("x"), py::arg("phases"),
        "Re<0|U(x)|0> -- the achieved QSP polynomial at x for the given phases.");
}
