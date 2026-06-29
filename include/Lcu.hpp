#pragma once
// Linear Combination of Unitaries (LCU): decompose a matrix into a weighted sum
// of Pauli strings, A = sum_j coef_j * P_j.

#include <complex>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "Common.hpp"

namespace qsvt {

class Lcu {
public:
    /// @param A The matrix to decompose. Its dimension must be a power of two.
    explicit Lcu(const Matrix& A);

    /// Build the full set of n-qubit Pauli strings (4^n of them).
    void generate_pauli_strings();

    /// Compute the coefficient coef_j = (1/2^n) * tr(P_j A) for each string.
    /// Requires generate_pauli_strings() to have been called.
    void generate_coefs();

    const std::vector<Matrix>& get_pauli_strings() const { return pauli_strings_; }
    const std::vector<Complex>& get_coefs() const { return coefs_; }

    /// Rebuild A = sum_j coef_j * P_j (for verification).
    Matrix reconstruct() const;

private:
    int n_qubits_{0};
    int n_pauli_strings_{0};
    std::vector<Matrix> pauli_strings_;
    std::vector<Eigen::Matrix2cd> pauli_matrices_;
    std::vector<Complex> coefs_;
    Matrix A_;
};

} // namespace qsvt
