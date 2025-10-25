#include <gtest/gtest.h>
#include <qrack/qfactory.hpp> // Include the Qrack header

using namespace Qrack; // Use the Qrack namespace

// Test 1: Check the probability *before* measurement
// This is the best way to test if the X gate worked.
TEST(QrackGates, XGateSetsProbToOne) {
    // Arrange: Set up a 2-qubit register in state |00>
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 2, ZERO_BCI);

    // Act: Apply an X gate to qubit 0
    qReg->X(0); // State should now be |10>

    // Assert: Check the probability of qubit 0 being |1>
    // We use EXPECT_DOUBLE_EQ for floating-point numbers.
    EXPECT_DOUBLE_EQ(qReg->Prob(0), 1.0);
}

// Test 2: Check the *result* of a measurement
// This tests that the M() function correctly returns the collapsed state.
TEST(QrackMeasure, MeasureReturnsCorrectResult) {
    // Arrange: Set up a 2-qubit register in state |00>
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 2, ZERO_BCI);

    // Act: Apply an X gate
    qReg->X(0); // State should be |10>

    // Assert: Measure qubit 0.
    // The M() function returns a bool: true for |1>, false for |0>.
    // Since the state is |1>, we expect the result to be true.
    EXPECT_TRUE(qReg->M(0));
}

// Test 3: Your original test (Prob *after* measurement)
// This is also a valid test! It confirms that after
// measuring |1>, the state *stays* |1>.
TEST(QrackMeasure, ProbAfterMeasureIsOne) {
    // Arrange
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 2, ZERO_BCI);

    // Act
    qReg->X(0);
    qReg->M(0); // Measure and collapse to |1>

    // Assert
    // The probability of measuring |1> *again* should be 1.0
    EXPECT_DOUBLE_EQ(qReg->Prob(0), 1.0);
}