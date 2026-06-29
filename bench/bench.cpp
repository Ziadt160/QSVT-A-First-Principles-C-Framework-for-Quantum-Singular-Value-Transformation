// Benchmark harness: emits the test inputs (random unitaries, target
// polynomials) together with this framework's results to a JSON file, which the
// companion benchmark.py feeds to Qiskit / PennyLane for an apples-to-apples
// comparison on identical inputs.

#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "QspAngleSolver.hpp"
#include "ShannonDecomposition.hpp"

using namespace qsvt;
using Clock = std::chrono::steady_clock;

namespace {

Matrix sampleUnitary(Eigen::Index dim, unsigned seed)
{
    Matrix m(dim, dim);
    unsigned state = seed * 2654435761u + 1u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state) / 4294967296.0 - 0.5;
    };
    for (Eigen::Index r = 0; r < dim; ++r)
        for (Eigen::Index c = 0; c < dim; ++c)
            m(r, c) = Complex(next(), next());
    Eigen::HouseholderQR<Matrix> qr(m);
    return Matrix(qr.householderQ());
}

double evalPoly(const std::vector<double>& coeffs, double x)
{
    double acc = 0.0;
    for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
        acc = acc * x + *it;
    }
    return acc;
}

double medianMicros(const std::vector<double>& v)
{
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    const std::string outPath = (argc > 1) ? argv[1] : "bench_input.json";
    std::ofstream o(outPath);
    o << std::setprecision(17);
    o << "{\n";

    // -- Decomposition: random n-qubit unitaries ---------------------------
    o << "  \"unitaries\": [\n";
    const std::vector<int> ns = {1, 2, 3, 4, 5};
    for (std::size_t idx = 0; idx < ns.size(); ++idx) {
        const int n = ns[idx];
        const Eigen::Index dim = Eigen::Index(1) << n;
        const Matrix U = sampleUnitary(dim, 100u + n);

        // Time the decomposition (median of a few runs); measure once for gates.
        std::vector<double> times;
        std::size_t gateCount = 0, cnotCount = 0;
        double reconErr = 0.0;
        const int repeats = (n <= 3) ? 20 : 5;
        for (int rpt = 0; rpt < repeats; ++rpt) {
            const auto t0 = Clock::now();
            ShannonDecomposition sd(U);
            const auto t1 = Clock::now();
            times.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
            if (rpt == 0) {
                gateCount = sd.gates().size();
                for (const auto& g : sd.gates())
                    if (g.isCnot) ++cnotCount;
                reconErr = (sd.circuitMatrix() - U).norm();
            }
        }

        o << "    {\"n\": " << n << ", \"dim\": " << dim
          << ", \"ours\": {\"gates\": " << gateCount
          << ", \"cnots\": " << cnotCount
          << ", \"recon_err\": " << reconErr
          << ", \"time_us\": " << medianMicros(times) << "},\n";

        auto writeMat = [&](const char* key, bool imag) {
            o << "     \"" << key << "\": [";
            for (Eigen::Index r = 0; r < dim; ++r) {
                o << (r ? ", [" : "[");
                for (Eigen::Index c = 0; c < dim; ++c) {
                    const double v = imag ? U(r, c).imag() : U(r, c).real();
                    o << (c ? ", " : "") << v;
                }
                o << "]";
            }
            o << "]";
        };
        writeMat("real", false);
        o << ",\n";
        writeMat("imag", true);
        o << "}";
        o << (idx + 1 < ns.size() ? ",\n" : "\n");
    }
    o << "  ],\n";

    // -- QSP angle finding: target polynomials (monomial coefficients) ------
    // Each is a Chebyshev combination with sum|c_k| <= 0.9, so |p(x)| <= 1.
    struct QspCase { int degree; std::vector<double> coeffs; };
    const std::vector<QspCase> qspCases = {
        {1, {0.0, 0.6}},                          // 0.6 T1
        {2, {-0.5, 0.0, 1.0}},                    // 0.5 T2
        {3, {0.0, -1.1, 0.0, 2.0}},               // 0.4 T1 + 0.5 T3
        {4, {0.3, 0.0, -2.6, 0.0, 3.2}},          // 0.2 T0 + 0.3 T2 + 0.4 T4
        {5, {0.0, 0.1, 0.0, -0.8, 0.0, 1.6}},     // 0.5 T1 + 0.3 T3 + 0.1 T5
    };

    o << "  \"qsp\": [\n";
    for (std::size_t idx = 0; idx < qspCases.size(); ++idx) {
        const auto& qc = qspCases[idx];
        QspAngleSolver solver(qc.degree);

        std::vector<double> times;
        QspSolveResult res;
        for (int rpt = 0; rpt < 5; ++rpt) {
            const auto t0 = Clock::now();
            res = solver.solve([&](double x) { return evalPoly(qc.coeffs, x); });
            const auto t1 = Clock::now();
            times.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        o << "    {\"degree\": " << qc.degree << ", \"coeffs\": [";
        for (std::size_t i = 0; i < qc.coeffs.size(); ++i)
            o << (i ? ", " : "") << qc.coeffs[i];
        o << "], \"ours\": {\"residual\": " << res.residual
          << ", \"time_us\": " << medianMicros(times)
          << ", \"converged\": " << (res.converged ? "true" : "false")
          << ", \"phases\": [";
        for (std::size_t i = 0; i < res.phases.size(); ++i)
            o << (i ? ", " : "") << res.phases[i];
        o << "]}}";
        o << (idx + 1 < qspCases.size() ? ",\n" : "\n");
    }
    o << "  ]\n}\n";

    o.close();
    std::cout << "Wrote benchmark inputs + our results to " << outPath << "\n";
    return 0;
}
