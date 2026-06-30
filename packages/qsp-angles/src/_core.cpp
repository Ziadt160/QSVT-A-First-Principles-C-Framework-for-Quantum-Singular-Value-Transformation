// pybind11 module for qsp-angles: exposes only the standalone QSP angle solver.
//
// The heavy QSVT/Qrack stack is intentionally NOT here -- this package is the
// extracted angle-finding wedge, which depends on nothing but Eigen + STL.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "QspAngleSolver.hpp"
#include "SymQspAngleSolver.hpp"

namespace py = pybind11;
using namespace qsvt;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Fast C++ QSP angle solvers: a symmetric-QSP Newton method "
              "(sym_qsp, default) and a homotopy-continuation fallback.";

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
        "than poly_to_angles. Returns {phases, residual, converged}.");

    m.def(
        "response",
        [](double x, const std::vector<double>& phases) {
            return QspAngleSolver::response(x, phases);
        },
        py::arg("x"), py::arg("phases"),
        "Re<0|U(x)|0> -- the achieved QSP polynomial at x for the given phases.");
}
