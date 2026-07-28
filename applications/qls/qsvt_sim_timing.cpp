// Wall-clock of actually simulating the compiled QSVT inversion circuit on the
// Qrack backend (which runs on the GTX 1650 / CUDA on this machine). Answers
// "does the GPU help?" empirically, alongside resource_scaling.cpp's gate counts.
//
// Link against the framework + Qrack (see build_qrack/src/.../link.txt):
//   g++ -O3 -std=c++17 -Iinclude -I/usr/include/eigen3 -I/usr/local/include \
//       applications/qls/qsvt_sim_timing.cpp build_qrack/src/libqsvt_lib.a \
//       /usr/local/lib/qrack/libqrack.a /usr/lib/x86_64-linux-gnu/libcudart.so \
//       -lpthread -o qsvt_sim_timing && ./qsvt_sim_timing

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "Gate.hpp"         // runCircuit, Gate
#include "QsvtPipeline.hpp" // compileMatrixFunction, QsvtProgram
#include "Qsvt.hpp"         // QSVT engine (owns the Qrack simulator)

using namespace qsvt;
using clk = std::chrono::high_resolution_clock;

// One n per process: creating multiple Qrack CUDA simulators in a single
// process segfaults (a Qrack context-reuse issue), so loop in the shell:
//   for n in 1 2 3 4 5; do ./qsvt_sim_timing $n; done
int main(int argc, char** argv)
{
    std::srand(42);
    const double delta = 0.3;
    const double eps = (delta / 4.0) * (delta / 4.0);
    const double c = 0.9 * 2.0 * std::sqrt(eps);
    auto invTarget = [&](double x) { return c * x / (x * x + eps); };
    const int degree = 25;
    const int nsys = argc > 1 ? std::atoi(argv[1]) : 1;

    {
        const Eigen::Index k = Eigen::Index(1) << nsys;
        Eigen::VectorXd ev(k);
        for (Eigen::Index i = 0; i < k; ++i) ev(i) = delta + (1.0 - delta) * double(i) / double(k - 1);
        Matrix G = Matrix::Random(k, k);
        Eigen::HouseholderQR<Matrix> qr(G);
        Matrix Wu = qr.householderQ();
        Matrix A = Wu * ev.cast<Complex>().asDiagonal() * Wu.adjoint();
        A = (A + A.adjoint()) * 0.5;

        const auto tb0 = clk::now();
        const QsvtProgram prog = compileMatrixFunction(A, invTarget, degree);
        const auto tb1 = clk::now();

        // Simulate the gate circuit on the Qrack (GPU) backend.
        QSVT engine(nsys); // nsys system + 1 ancilla, on the default CUDA device
        auto sim = engine.get_simulator();
        std::vector<bitLenInt> qubits(nsys + 1);
        std::iota(qubits.begin(), qubits.end(), bitLenInt(0));

        const auto ts0 = clk::now();
        runCircuit(prog.circuit, sim, qubits);
        volatile double force = sim->Prob(0); // force the backend to evaluate
        (void)force;
        const auto ts1 = clk::now();

        std::printf("%d,%d,%d,%.0f,%.1f\n", nsys, prog.numQubits, prog.resources.cnot,
                    std::chrono::duration<double, std::milli>(tb1 - tb0).count(),
                    std::chrono::duration<double, std::milli>(ts1 - ts0).count());
        std::fflush(stdout);
    }
    return 0;
}
