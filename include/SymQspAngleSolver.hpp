#pragma once
// Symmetric-QSP angle finding via the Dong-Lin-Ni-Wang robust Newton method
// (arXiv:2307.12468), the algorithm behind pyqsp's `sym_qsp`. Validated against
// pyqsp to ~15 digits and benchmarked to degree > 1000.
//
// Like `QspAngleSolver`, this finds a symmetric phase sequence Phi = (phi_0,
// ..., phi_d) for a real target polynomial f(x) of definite parity (d mod 2)
// with |f(x)| <= 1 on [-1, 1], such that, in the Wx convention,
//
//   U(x, Phi) = e^{i phi_0 Z} * prod_{k=1..d} [ W(x) e^{i phi_k Z} ],
//   W(x) = e^{i arccos(x) X} = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]],
//
// satisfies  Re<0|U(x, Phi)|0> = f(x)  (the SAME Re convention as
// QspAngleSolver). Internally the Newton iteration targets the IMAGINARY part of
// the response; `reconstructFull` then applies a -pi/4 shift to the first and
// last full phase to convert back to the Re convention, so the returned phases
// drop straight into `QspAngleSolver::response`.
//
// Compared to the homotopy solver this is strictly faster, converges in ~4-5
// Newton iterations at any degree, and reaches degree > 1000 at machine
// precision -- so it is the preferred default. It is, like QspAngleSolver, a
// self-contained Eigen + STL solver embeddable in a compiled stack.

#include <functional>

#include "QspAngleSolver.hpp" // QspSolveResult (shared return type)

namespace qsvt {

class SymQspAngleSolver {
public:
    /// @param degree Degree d of the target polynomial.
    explicit SymQspAngleSolver(int degree) : degree_(degree) {}

    /// Find phases approximating @p target. @p target must be a degree-<=d
    /// polynomial of parity (d mod 2) with magnitude <= 1 on [-1, 1]. The
    /// target is in the Re convention (Re<0|U|0> = f), same as QspAngleSolver.
    ///
    /// Returns the full (length d+1) symmetric phases in the Re convention, the
    /// worst-case grid residual max|Re<0|U|0> - f| over [-1, 1] (measured with
    /// QspAngleSolver::response), iteration count, and converged = residual <
    /// 1e-6.
    QspSolveResult solve(const std::function<double(double)>& target) const;

    int degree() const { return degree_; }

private:
    int degree_;
};

} // namespace qsvt
