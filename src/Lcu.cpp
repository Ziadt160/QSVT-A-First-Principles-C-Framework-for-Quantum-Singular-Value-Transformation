#include "Lcu.hpp"

#include <cmath>

#include <eigen3/unsupported/Eigen/KroneckerProduct>

namespace qsvt {

namespace {
const Complex kI(0.0, 1.0);
}

Lcu::Lcu(const Matrix& A)
    : A_(A)
{
    Eigen::Matrix2cd I, X, Y, Z;
    I << 1, 0,
         0, 1;
    X << 0, 1,
         1, 0;
    Y << 0, -kI,
         kI, 0;
    Z << 1, 0,
         0, -1;

    pauli_matrices_ = {I, X, Y, Z};

    const Eigen::Index shape = A_.cols();
    n_qubits_ = static_cast<int>(std::lround(std::log2(static_cast<double>(shape))));
    n_pauli_strings_ = static_cast<int>(std::lround(std::pow(4.0, n_qubits_)));
}

void Lcu::generate_pauli_strings()
{
    pauli_strings_.clear();
    pauli_strings_.reserve(static_cast<std::size_t>(n_pauli_strings_));

    for (int i = 0; i < n_pauli_strings_; ++i) {
        if (n_qubits_ == 0) {
            Matrix p(1, 1);
            p(0, 0) = 1;
            pauli_strings_.push_back(p);
            continue;
        }

        int temp_index = i;
        Matrix current = pauli_matrices_[temp_index % 4];
        temp_index /= 4;

        for (int j = 1; j < n_qubits_; ++j) {
            const Matrix next = pauli_matrices_[temp_index % 4];
            current = Eigen::kroneckerProduct(next, current).eval();
            temp_index /= 4;
        }

        pauli_strings_.push_back(current);
    }
}

void Lcu::generate_coefs()
{
    coefs_.clear();
    coefs_.reserve(pauli_strings_.size());

    const Complex multiplier = 1.0 / std::pow(2.0, n_qubits_);
    for (const auto& pauli_string : pauli_strings_) {
        const Matrix product = pauli_string * A_;
        coefs_.push_back(multiplier * product.trace());
    }
}

Matrix Lcu::reconstruct() const
{
    Matrix result = Matrix::Zero(A_.rows(), A_.cols());
    for (std::size_t j = 0; j < pauli_strings_.size(); ++j) {
        result += coefs_[j] * pauli_strings_[j];
    }
    return result;
}

} // namespace qsvt
