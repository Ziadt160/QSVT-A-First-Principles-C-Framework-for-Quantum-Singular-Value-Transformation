#pragma once
// VENDORED from the QSVT_Project root (include/SymQspAngleSolver.hpp).
// This file has no third-party-library dependency; it is byte-for-byte
// identical to the canonical source apart from this vendored comment block.
// Keep in sync with the root.
//
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
// precision -- so it is the preferred default. It is a self-contained,
// ZERO-dependency core: SymQspAngleSolver.{hpp,cpp} use only the C++ standard
// library (no linear-algebra or FFT third-party library), so the two files drop
// straight into any compiled stack with nothing to install.

#include <functional>
#include <vector>

#include "QspResult.hpp" // QspSolveResult (shared return type)

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
    /// this solver's own self-contained response()), iteration count, and
    /// converged = residual < 1e-6.
    QspSolveResult solve(const std::function<double(double)>& target) const;

    /// Re<0|U(x, phases)|0> -- the achieved QSP polynomial in the Wx convention.
    /// A self-contained 2x2 complex matrix product (standard library only, no
    /// dependency on QspAngleSolver), numerically equivalent to
    /// QspAngleSolver::response so the phases this solver returns can be graded
    /// interchangeably.
    static double response(double x, const std::vector<double>& phases);

    int degree() const { return degree_; }

private:
    int degree_;
};

} // namespace qsvt
