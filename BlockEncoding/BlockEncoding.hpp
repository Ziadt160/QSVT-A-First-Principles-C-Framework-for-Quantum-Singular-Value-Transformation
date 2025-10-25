#pragma once // Prevents the header from being included multiple times
#include <qrack/qfactory.hpp>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <stdexcept>

using namespace Qrack;

typedef Eigen::Matrix<Qrack::complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DynamicMatrix;

class BlockEncodingException : public std::runtime_error {
public:
    explicit BlockEncodingException(const std::string& message)
        : std::runtime_error("Block Encoding Error: " + message) {}
};

class BlockEncoding {
public:
    /**
     * @brief Prepares a block encoding for a scalar 'a'.
     * @param qReg The quantum simulator interface.
     * @param scalar The scalar to encode. Must be in the range [-1, 1].
     */
    BlockEncoding(Qrack::QInterfacePtr qReg, double scalar);

    /**
     * @brief Prepares a block encoding for a matrix 'A'.
     * @param qReg The quantum simulator interface.
     * @param matrix The matrix to encode. The matrix norm |A| must be <= 1.
     */
    BlockEncoding(Qrack::QInterfacePtr qReg, const DynamicMatrix& matrix);

    /**
     * @brief Applies the block encoding unitary U to a target qubit.
     */
    void apply(bitLenInt arget) const;

    /**
     * @brief Applies the adjoint (inverse) of the block encoding unitary.
     */
    void apply_adjoint(bitLenInt target) const;

    /**
     * @brief Applies the controlled block encoding unitary C-U.
     * @param controls A vector of control qubit IDs.
     * @param target The target qubit ID.
     */
    void controlled_apply(const std::vector<uint8_t>& controls, bitLenInt target) const;

    /**
     * @brief Applies the anti-controlled block encoding unitary C-U.
     * @param controls A vector of control qubit IDs.
     * @param target The target qubit ID.
     */
    void anti_controlled_apply(const std::vector<uint8_t>& controls, bitLenInt target) const;

    const DynamicMatrix& get_encoded_matrix() const;


private:
    Qrack::QInterfacePtr qReg;
    double theta;
    DynamicMatrix matrix;
};