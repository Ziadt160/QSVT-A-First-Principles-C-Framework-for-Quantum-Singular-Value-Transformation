#include "Lcu.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>

const std::complex<double> i(0.0, 1.0);


Lcu::Lcu(const MatrixXcd& A)
{
    this->A = A;

    Matrix2cd I;
    I << 1, 0,
        0, 1;

    Matrix2cd X;
    X << 0, 1,
        1, 0;
    Matrix2cd Y;
    Y << 0, -i,
        i,  0;

    Matrix2cd Z;
    Z << 1,  0,
        0, -1;


    pauli_matricies.push_back(I);
    pauli_matricies.push_back(X);
    pauli_matricies.push_back(Y);
    pauli_matricies.push_back(Z);


    int shape = A.cols();

    n_qubits = static_cast<int>(std::round(std::log2(shape)));

    n_pauli_strings = std::pow(4 , n_qubits);
}

void Lcu::generate_pauli_strings()
{
    for (int i = 0; i < n_pauli_strings; i += 1)
    {
        int temp_index = i;

        if (n_qubits == 0) {
            MatrixXcd p(1,1); p(0,0) = 1;
            pauli_strings.push_back(p);
            continue;
        }        

        MatrixXcd current = pauli_matricies[ temp_index % 4];

        temp_index = temp_index / 4;

        for(int j = 1; j < n_qubits; j += 1)
        {
            MatrixXcd next = pauli_matricies[ temp_index % 4];

            current = kroneckerProduct(next, current);

            temp_index = temp_index / 4;
        }

        pauli_strings.push_back(current);
    };
};

void Lcu::generate_coefs()
{
    this->coefs.clear();
    this->coefs.reserve(this->pauli_strings.size());

    for( auto const& pauli_string : pauli_strings)
    {
        MatrixXcd matrix = pauli_string * A;
        dcomplex multiplier = 1.0 / std::pow(2.0, n_qubits);

        coefs.push_back(  multiplier * matrix.trace());
    }
}

const std::vector<MatrixXcd>& Lcu::get_pauli_strings() const
{
    return this->pauli_strings;
}

const std::vector<std::complex<double>>& Lcu::get_coefs() const
{
    return this->coefs;
}