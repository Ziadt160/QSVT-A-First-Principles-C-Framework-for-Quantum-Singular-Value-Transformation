#pragma once
// Shared result type for the QSP angle solvers. Kept in its own zero-dependency
// header (just <vector>) so the symmetric-QSP Newton core
// (SymQspAngleSolver.{hpp,cpp}) is a self-contained, dependency-free two-file
// drop-in: it returns this struct without pulling in QspAngleSolver.hpp or any
// third-party library. Both QspAngleSolver and SymQspAngleSolver include this.

#include <vector>

namespace qsvt {

struct QspSolveResult {
    std::vector<double> phases; ///< length d+1
    double residual{0.0};       ///< max |Re<0|U|0> - f| over a fine grid
    bool converged{false};
    int iterations{0};
};

} // namespace qsvt
