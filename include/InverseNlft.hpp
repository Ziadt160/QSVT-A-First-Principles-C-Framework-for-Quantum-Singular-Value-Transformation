#pragma once
// Inverse nonlinear Fourier transform by the Half-Cholesky method
// (Ni-Ying, arXiv:2410.06409) -- step 4 of the NLFT port
// (packages/qsp-angles/docs/NLFT_PORT_PLAN.md).
//
// Given the target b and c ~ b/a from Weiss completion, recovers the sequence F
// whose NLFT is (a, b). The QSP phase factors follow from F.
//
// Implementation note: the reference does a QR of a 2 x m matrix at every one of
// the n steps. A 2 x m QR is a single Givens rotation, so it is written here in
// closed form -- no linear-algebra dependency, and O(m) per step, O(n^2) total.
// With a = conj(u_0), b = conj(v_0) and nrm = sqrt(|u_0|^2 + |v_0|^2):
//
//     u'_j = ( conj(u_0) u_j + conj(v_0) v_j ) / nrm
//     v'_j = ( -v_0 u_j     + u_0 v_j       ) / nrm
//
// The Householder/Givens phase convention is irrelevant to the result because
// each stored column is normalised by its own leading entry.

#include <vector>

#include "LaurentPoly.hpp"

namespace qsvt {

/// A finitely supported complex sequence over Z, as in nlft-qsp's
/// NonLinearFourierSequence: element k is coeffs[k - supportStart].
struct NlftSequence {
    std::vector<Complexd> coeffs;
    int supportStart{0};
};

/// Inverse NLFT of (b, c) via Half-Cholesky.
/// @param b the target polynomial.
/// @param c an approximation of b/a, whose support must end where b's does
///          (produced by weissComplete(..., withRatio = true)).
NlftSequence inverseNlft(const LaurentPoly& b, const LaurentPoly& c);

/// The Half-Cholesky columns of L for I + B B^dag = L D L^dag.
/// Column k has length (n+1)-k; entry j of column k is L[k+j][k].
/// Exposed for validation against the reference implementation.
std::vector<std::vector<Complexd>> halfCholeskyColumns(std::vector<Complexd> u,
                                                       std::vector<Complexd> v);

} // namespace qsvt
