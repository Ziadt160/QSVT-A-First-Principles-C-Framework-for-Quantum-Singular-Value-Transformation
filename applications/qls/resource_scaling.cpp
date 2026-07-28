// W3 / simulability-ceiling spike: how the compiled QSVT inversion circuit's
// gate count grows with system size. This is the evidence for WHY a GPU
// statevector simulator does NOT unlock large n for a DENSE block-encoding:
// the per-U_W gate count is ~0.58*4^(n+1) (Shannon decomposition of a generic
// 2^(n+1) unitary), so the circuit becomes impossible to even build long before
// the statevector becomes hard to store. Gate-count wall, not statevector wall.
//
// Resource counting needs only gate synthesis (Eigen) -- no Qrack, no simulator.
//
//   g++ -O3 -std=c++17 -Iinclude -I/usr/include/eigen3 applications/qls/resource_scaling.cpp \
//       src/QsvtPipeline.cpp src/QspAngleSolver.cpp src/ShannonDecomposition.cpp \
//       src/CSDecomposition.cpp src/TwoQubitSynthesis.cpp src/KAKDecomposition.cpp \
//       src/Circuit.cpp -o resource_scaling && ./resource_scaling

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <eigen3/Eigen/Dense>

#include "QsvtPipeline.hpp"

using namespace qsvt;
using clk = std::chrono::high_resolution_clock;

int main()
{
    std::srand(42);
    const double delta = 0.3; // condition number ~ 1/delta ~ 3.3
    const double eps = (delta / 4.0) * (delta / 4.0);
    const double c = 0.9 * 2.0 * std::sqrt(eps);
    auto invTarget = [&](double x) { return c * x / (x * x + eps); };
    const int degree = 25;

    std::printf("sys_qubits,total_qubits,cnots,gates,depth,compile_s\n");
    for (int nsys = 1; nsys <= 6; ++nsys) {
        const Eigen::Index k = Eigen::Index(1) << nsys; // 2^nsys system dimension

        // Generic dense Hermitian A with spectrum in [delta, 1] (worst-case for
        // synthesis: a full block-encoding, no sparsity to exploit).
        Eigen::VectorXd ev(k);
        for (Eigen::Index i = 0; i < k; ++i) ev(i) = delta + (1.0 - delta) * double(i) / double(k - 1);
        Matrix G = Matrix::Random(k, k);
        Eigen::HouseholderQR<Matrix> qr(G);
        Matrix Wu = qr.householderQ();
        Matrix A = Wu * ev.cast<Complex>().asDiagonal() * Wu.adjoint();
        A = (A + A.adjoint()) * 0.5;

        const auto t0 = clk::now();
        const QsvtProgram prog = compileMatrixFunction(A, invTarget, degree);
        const auto t1 = clk::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        std::printf("%d,%d,%d,%d,%d,%.2f\n", nsys, prog.numQubits,
                    prog.resources.cnot, prog.resources.total, prog.resources.depth, secs);
        std::fflush(stdout);
    }
    return 0;
}
