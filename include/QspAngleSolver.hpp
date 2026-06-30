#pragma once
// Quantum Signal Processing angle finding.
//
// Given a real target polynomial f(x) of degree d, definite parity (d mod 2),
// and |f(x)| <= 1 on [-1, 1], find a (symmetric) phase sequence Phi = (phi_0,
// ..., phi_d) such that, in the Wx convention,
//
//   U(x, Phi) = e^{i phi_0 Z} * prod_{k=1..d} [ W(x) e^{i phi_k Z} ],
//   W(x) = e^{i arccos(x) X} = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]],
//
// satisfies  Re<0|U(x, Phi)|0> = f(x).
//
// The solver uses the symmetric ansatz (phi_k = phi_{d-k}) and a damped
// Gauss-Newton (Levenberg-Marquardt) fit over Chebyshev nodes. The resulting
// phases can drive the `Qsp` class with the signal unitary W(x).

#include <complex>
#include <functional>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "QspResult.hpp" // QspSolveResult (shared return type)

namespace qsvt {

class QspAngleSolver {
public:
    /// @param degree Degree d of the target polynomial.
    explicit QspAngleSolver(int degree) : degree_(degree) {}

    /// Find phases approximating @p target. @p target must be a degree-<=d
    /// polynomial of parity (d mod 2) with magnitude <= 1 on [-1, 1].
    QspSolveResult solve(const std::function<double(double)>& target) const;

    /// The full QSP unitary U(x, phases) (Wx convention).
    static Eigen::Matrix2cd unitary(double x, const std::vector<double>& phases);

    /// Full (0,0) entry of U(x, phases).
    static std::complex<double> response00(double x,
                                           const std::vector<double>& phases)
    {
        return unitary(x, phases)(0, 0);
    }

    /// Re<0|U(x, phases)|0> -- the achieved QSP polynomial.
    static double response(double x, const std::vector<double>& phases)
    {
        return response00(x, phases).real();
    }

    int degree() const { return degree_; }

private:
    int degree_;
};

} // namespace qsvt
