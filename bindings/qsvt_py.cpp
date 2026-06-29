// Python bindings for the C++ QSVT toolkit (pybind11).
//
// Exposes the matrix-function -> circuit compilation stack to Python: QSP angle
// finding, unitary decompositions (with OpenQASM + resource estimates), the
// full QSVT pipeline, and the applications (matrix inversion via
// compile_matrix_function, Hamiltonian simulation, eigenvalue thresholding).
// Eigen matrices convert to/from NumPy automatically.

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "Common.hpp"
#include "EigenvalueThreshold.hpp"
#include "Gate.hpp"
#include "HamiltonianSimulation.hpp"
#include "Lcu.hpp"
#include "QspAngleSolver.hpp"
#include "QsvtPipeline.hpp"
#include "ShannonDecomposition.hpp"
#include "TwoQubitSynthesis.hpp"

namespace py = pybind11;
using namespace qsvt;

namespace {

py::dict decompReport(const std::vector<Gate>& gates, int nQubits, const Matrix& U)
{
    const ResourceCounts rc = countResources(gates, nQubits);
    py::dict d;
    d["num_qubits"] = nQubits;
    d["cnot"] = rc.cnot;
    d["single_qubit"] = rc.singleQubit;
    d["total"] = rc.total;
    d["depth"] = rc.depth;
    d["qasm"] = toQasm(gates, nQubits);
    d["reconstruction_error"] = (denseCircuit(gates, nQubits) - U).norm();
    return d;
}

} // namespace

PYBIND11_MODULE(qsvt_native, m)
{
    m.doc() = "C++ QSVT toolkit: matrix-function -> quantum-circuit compilation";

    // --- QSP angle finding ------------------------------------------------
    m.def(
        "poly_to_angles",
        [](const std::function<double(double)>& target, int degree) {
            const QspSolveResult r = QspAngleSolver(degree).solve(target);
            py::dict d;
            d["phases"] = r.phases;
            d["residual"] = r.residual;
            d["converged"] = r.converged;
            return d;
        },
        py::arg("target"), py::arg("degree"),
        "QSP phases realising a degree-d polynomial approximation of target(x).");

    m.def(
        "qsp_response",
        [](double x, const std::vector<double>& phases) {
            return QspAngleSolver::response(x, phases);
        },
        py::arg("x"), py::arg("phases"),
        "Re<0|U(x)|0> -- the achieved QSP polynomial at x.");

    // --- Unitary decomposition -> native gates / QASM / resources ---------
    m.def(
        "shannon_decompose",
        [](const Matrix& U) {
            ShannonDecomposition s(U);
            return decompReport(s.gates(), s.numQubits(), U);
        },
        py::arg("U"),
        "Quantum Shannon Decomposition of an n-qubit unitary; returns resource "
        "counts, OpenQASM, and reconstruction error.");

    m.def(
        "two_qubit_synthesize",
        [](const Eigen::Matrix4cd& U) {
            TwoQubitSynthesis s(U);
            return decompReport(s.gates(), 2, Matrix(U));
        },
        py::arg("U"), "KAK-based 2-qubit synthesis -> gates / QASM / resources.");

    // --- QSVT pipeline ----------------------------------------------------
    m.def("rotation_block_encoding", &rotationBlockEncoding, py::arg("A"),
          "Rotation-convention block-encoding U_W of a Hermitian contraction A.");
    m.def("qsvt_unitary", &qsvtUnitary, py::arg("A"), py::arg("phases"),
          "Dense QSVT operator U_Phi.");
    m.def("qsvt_block", &qsvtBlock, py::arg("A"), py::arg("phases"),
          "Top-left block of the QSVT operator = P(A).");

    m.def(
        "compile_matrix_function",
        [](const Matrix& A, const std::function<double(double)>& target, int degree) {
            const QsvtProgram p = compileMatrixFunction(A, target, degree);
            py::dict d;
            d["phases"] = p.phases;
            d["num_qubits"] = p.numQubits;
            d["cnot"] = p.resources.cnot;
            d["total"] = p.resources.total;
            d["depth"] = p.resources.depth;
            d["poly_residual"] = p.polyResidual;
            d["converged"] = p.converged;
            d["qasm"] = toQasm(p.circuit, p.numQubits);
            return d;
        },
        py::arg("A"), py::arg("target"), py::arg("degree"),
        "Compile target(A) into a QSVT circuit (phases + QASM + resources).");

    // --- Applications -----------------------------------------------------
    m.def(
        "hamiltonian_evolution",
        [](const Matrix& H, double t, int degree) {
            const HamSimProgram prog = compileHamiltonianSimulation(H, t, degree);
            return simulatedEvolution(H, prog); // dense e^{-iHt} approximation
        },
        py::arg("H"), py::arg("t"), py::arg("degree"),
        "QSVT approximation of the time-evolution operator e^{-iHt}.");

    m.def(
        "spectral_projector",
        [](const Matrix& H, double mu, double w, int degree) {
            const ThresholdProgram prog =
                compileEigenvalueThreshold(H, mu, w, degree);
            return spectralProjector(H, prog);
        },
        py::arg("H"), py::arg("mu") = 0.0, py::arg("w") = 0.1, py::arg("degree") = 25,
        "Projector onto eigenvalues of H above mu, via QSVT sign(H-mu).");

    // --- LCU --------------------------------------------------------------
    m.def(
        "lcu_coefficients",
        [](const Matrix& A) {
            Lcu l(A);
            l.generate_pauli_strings();
            l.generate_coefs();
            return l.get_coefs();
        },
        py::arg("A"), "Pauli-string coefficients of A (LCU decomposition).");
}
