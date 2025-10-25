#pragma once

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>

using namespace Eigen;



class Lcu
{
private:

int n_qubits;
int n_pauli_strings;
std::vector<MatrixXcd> pauli_strings;
std::vector<Matrix2cd> pauli_matricies;
std::vector<std::complex<double>> coefs;
MatrixXcd A;

public:
    /**
     * @brief Constructor for LCU Decomposition
     * @param A the matrix that we want to decompose
     */
    explicit Lcu(const MatrixXcd& A);

    /**
     * @brief generating pauli strings
     */
    void generate_pauli_strings();

    /**
     * @brief generate the coefs for every combination of pauli strings
     */
    void generate_coefs();

    const std::vector<MatrixXcd>& get_pauli_strings() const;
    const std::vector<std::complex<double>>& get_coefs() const;
};