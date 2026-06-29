#pragma once
// KAK (Cartan) decomposition of a two-qubit unitary: U = K1 * A * K2, where
// K1, K2 are local (single-qubit tensor products) and A is the canonical
// entangling part, diagonal in the magic basis.

#include <stdexcept>
#include <string>
#include <tuple>

#include <eigen3/Eigen/Dense>

namespace qsvt {

class KAKException : public std::runtime_error {
public:
    explicit KAKException(const std::string& message)
        : std::runtime_error("KAK Decomposition Error: " + message) {}
};

/// Coefficients of the canonical interaction A = exp(i(ax XX + ay YY + az ZZ)).
/// These are derived from the magic-basis phases and are only defined up to the
/// (arbitrary) ordering of the eigenvalues returned by the solver.
struct AlphaCoefficients {
    double ax{0.0};
    double ay{0.0};
    double az{0.0};
};

class KAKDecomposition {
public:
    /// @param matrix The two-qubit unitary U in the computational basis.
    explicit KAKDecomposition(const Eigen::Matrix4cd& matrix);

    /// Factor U into (K1, A, K2). The product K1 * A * K2 reproduces U exactly.
    std::tuple<Eigen::Matrix4cd, Eigen::Matrix4cd, Eigen::Matrix4cd> solve();

    /// Canonical interaction angles, valid only after solve() has run.
    AlphaCoefficients getAMatrixAngles() const { return angles_; }

private:
    Eigen::Matrix4cd matrix_;
    Eigen::Matrix4cd Q_;        // computational -> magic basis change
    Eigen::Matrix4cd Q_dagger_;
    AlphaCoefficients angles_;
};

} // namespace qsvt
