# QSVT — A First-Principles C++ Framework for Quantum Singular Value Transformation

![tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![lang](https://img.shields.io/badge/C%2B%2B-17-00599C)

A from-scratch, **verified** C++ implementation of Quantum Singular Value
Transformation: it compiles a matrix function `f(A)` into a runnable quantum
circuit — block-encoding → QSP angle finding → native-gate synthesis → OpenQASM
and resource estimates — entirely in compiled C++ (Qrack GPU-simulator backend),
with optional Python bindings. Every component is checked against dense linear
algebra *and* the Qrack simulator.

Results at a glance
-------------------

- **Decomposition** (random unitary → native gates), verified vs Qiskit's QSD on
  identical inputs: reproduces the unitary to ~1e-13; **4 / 28 / 136 / 592 CNOTs**
  for n = 2..5 (~1.4x Qiskit's optimal QSD), and faster per-call once optimised.
- **QSP angle solver**: machine precision to **degree 100+** and near `|f| = 1`
  (homotopy continuation + analytic Jacobian) — degree 101 to ~1e-14 in ~1.7 s.
- **QSVT applications**, validated against exact linear algebra:
  - matrix inversion of a kappa~3 Hermitian to ~2% on its spectrum (degree-25 circuit),
  - Hamiltonian simulation `e^{-iHt}` to ~1e-13 (machine precision),
  - eigenvalue thresholding / spectral projection to ~5e-3.
- **Interop**: OpenQASM 2.0 export, CNOT/depth resource estimates, `qsvt_native`
  Python module (NumPy in/out).
- **29 GoogleTest cases** (dense + Qrack-simulator), benchmark harness vs
  Qiskit/PennyLane (`bench/`), CI building Qrack + the project + tests.

```bash
cmake -S . -B build && cmake --build build -j      # builds (Release by default)
ctest --test-dir build                              # 29 tests
./build/src/qsvt_app                                # demo (KAK, QSD, QSVT, inversion, e^{-iHt})
```

See `examples/qsvt_demo.py` for the Python API and [BENCHMARK.md](BENCHMARK.md)
for the full comparison.

Short project summary
---------------------

This repository implements Quantum Singular Value Transformation (QSVT) in C++
from the ground up. The code uses Qrack as the quantum simulator backend (Qrack can use CUDA to accelerate simulation). It implements the low-level building blocks (block-encoding, LCU, gate decompositions, QSP) and integrates them into a full QSVT pipeline with worked applications.

High-level roadmap (what this project is trying to do)
----------------------------------------------------

- Implement block-encoding primitives (encode scalars and small matrices into unitaries).
- Implement methods to decompose n-qubit unitaries into native gates as would be implemented on hardware (KAK, CS decomposition, etc.).
- Implement Linear Combination of Unitaries (LCU) support so we can represent matrices as weighted sums of Pauli strings.
- Implement Quantum Signal Processing (QSP) to apply polynomial transformations to singular values / eigenvalues.
- Combine the pieces to realize QSVT: prepare block-encodings, apply QSP-based polynomial filters, and extract transformed singular values.

Why this approach
------------------

There are two ways to make progress on block-encodings and QSVT: (1) use the statevector simulator to directly create large unitaries that act like the desired block-encoding (easier for simulation), or (2) implement block-encodings as they would be implemented on quantum hardware, by decomposing native gates and using controlled operations. This project chooses the second ("hardware-like") path because it leads to a better understanding of the gate-level construction and constraints for later compilation/optimization and for eventual runtime on real devices.

Current status (what's implemented)
-----------------------------------

All components live in the `qsvt` namespace. Shared types are centralised in
`include/Common.hpp` (double-precision Eigen aliases) and `include/QrackTypes.hpp`
(the Qrack-facing matrix type and a converter). A GoogleTest suite in `tests/`
covers round-trip correctness of every decomposition.

- BlockEncoding: scalar block-encoding (the reflection `[[a, q], [q, -a]]`) and
  matrix block-encoding via the standard dilation
  `U = [[A, sqrt(I - A A^H)], [sqrt(I - A^H A), -A^H]]`. The constructor stores
  the simulator handle and the resulting unitary. `apply(target)` runs the
  single-qubit case directly; `apply(qubits)` compiles a multi-qubit encoding to
  native gates (Quantum Shannon Decomposition) and runs them on the simulator.
- Lcu: generation of the Pauli-string basis and the coefficients
  `coef_j = (1/2^n) tr(P_j A)`, plus a `reconstruct()` helper that rebuilds the
  matrix from its Pauli decomposition.
- Qsp: a one-qubit QSP routine that interleaves a signal unitary with
  Z-rotations driven by a given angle set.
- QspAngleSolver: the QSP angle-finding routine. Given a real target polynomial
  f(x) (degree d, parity d mod 2, |f| <= 1), it finds a symmetric phase sequence
  with Re<0|U(x)|0> = f(x), using a damped Gauss-Newton (Levenberg-Marquardt)
  fit over Chebyshev nodes in the Wx convention, with an **exact analytic
  Jacobian** (prefix/suffix products of the QSP factors, O(d) per node) and a
  **homotopy continuation**: the target is morphed from T_d (which Phi = 0 solves
  exactly) to f, warm-starting each step. This reaches **degree 100+ near machine
  precision** (e.g. degree 101 to ~1e-14 in ~1.7 s; sub-200 ms through degree
  ~50) and handles targets near |f| = 1 -- where a cold-start solve stalls around
  degree 9. The resulting phases drive the `Qsp` class and the QSVT pipeline;
  verified against the forward model and on the Qrack simulator.
- KAKDecomposition: KAK (Cartan) decomposition for 2-qubit unitaries (SU(4)),
  returning local factors `K1`, `K2` and the canonical entangler `A`. The
  complex-symmetric (and unitary) matrix in the magic basis is diagonalised by
  simultaneous diagonalisation of its commuting real and imaginary parts
  (degenerate eigenspaces handled), with a parity correction that keeps `K1`,
  `K2` in `SO(4)` so they map to genuinely local gates. `K1 A K2` reproduces the
  input to machine precision.
- CSDecomposition: cosine-sine decomposition for any even-dimensional unitary
  (equal `n x n` bipartition), returning `L1, L2, R1, R2` and the rotation
  angles. This generalises beyond the 4x4-only KAK to arbitrary matrix sizes.
- TwoQubitSynthesis: compiles a 2-qubit unitary into a native gate sequence
  (single-qubit gates + CNOTs) via KAK — the local factors are split into
  single-qubit gates (Van Loan-Pitsianis) and the entangler is realised with
  `exp(i t ZZ) = CNOT (I (x) Rz(-2t)) CNOT` conjugated into XX/YY. Verified both
  against a dense reference model and by running on the Qrack simulator.
- ShannonDecomposition: compiles an arbitrary n-qubit unitary into native gates.
  It recurses the cosine-sine decomposition (split on the top qubit) into
  uniformly-controlled Ry/Rz rotations plus two demultiplexed (n-1)-qubit
  factors, bottoming out at a 2-qubit base case. The uniformly-controlled
  rotations use the Mottonen flat construction (2^k CNOTs), the 2-qubit base is
  a 4-CNOT synthesis (the XX/ZZ canonical rotations share a CNOT pair), and a
  unitary-preserving peephole pass cleans up (adjacent-CNOT cancellation +
  single-qubit fusion). This gives ~0.58*4^n CNOTs (4/28/136/592 for n=2..5) vs
  ~0.9*4^n naive. Verified dense and on the Qrack simulator up to 4-5 qubits;
  this is what `BlockEncoding::apply(qubits)` uses. (~1.4x above Qiskit's
  fully-optimised QSD — see next steps.)
- QSVT (engine): a wrapper around the Qrack interface that creates the simulator
  (system qubits plus one ancilla) and tracks the ancilla index.
- QsvtPipeline: the full Quantum Singular Value Transformation. For a Hermitian
  contraction `A` and QSP phases `Phi`, it builds the rotation-convention
  block-encoding `U_W = [[A, i sqrt(I-A^2)], [i sqrt(I-A^2), A]]` and the QSVT
  operator `U_Phi = E(phi_0) prod_k [U_W E(phi_k)]` (with `E(phi)=e^{i phi Z_anc}`),
  whose top-left block equals `P(A)` — verified to machine precision against the
  polynomial applied to A's eigenvalues. `qsvtCircuit` emits the whole thing as a
  native gate sequence (U_W compiled via ShannonDecomposition; projector
  rotations are ancilla Z-rotations), and `compileMatrixFunction(A, f, degree)`
  is the **matrix-function compiler**: target function -> QSP phases -> runnable
  QSVT circuit + resource profile. A worked matrix-inversion example lives in
  `main.cpp`.
- Resource estimation and QASM export (`countResources`, `toQasm` in
  `Gate.hpp` / `Circuit.cpp`): CNOT/single-qubit/depth counts and OpenQASM 2.0
  output (`cx` + `U(theta,phi,lambda)` via a ZYZ factorisation), so compiled
  circuits run on other toolchains / hardware.
- HamiltonianSimulation: a QSVT application implementing `e^{-iHt}` for a
  Hermitian `H` via `e^{-iHt} = cos(tH) - i sin(tH)`. It fits `cos(tx)` (even)
  and `sin(tx)` (odd) with the angle solver and applies them with QSVT; the
  Hermitian part of each block recovers the real matrix function. On a 2-qubit
  model at `t = 3`, degree-21 QSVT reproduces `e^{-iHt}` to ~1e-14 (machine
  precision -- the truncation falls below it), with ~560/588-CNOT cos/sin
  circuits. Verified against the exact evolution operator.
- EigenvalueThreshold: spectral projection / ground-state filtering, the third
  canonical QSVT application. Applies a smooth `sign(H - mu)` (odd `erf(x/w)`
  target) to project onto eigenvalues above a threshold: `Pi = (I + sign(H-mu))/2`.
  Projects a 4-qubit-block model's positive eigenspace to ~5e-3 at degree 25.

Files of interest
-----------------

- `include/Common.hpp` — shared double-precision Eigen type aliases and tolerance.
- `include/QrackTypes.hpp` — Qrack-facing matrix type and `toQMatrix` converter.
- `include/BlockEncoding.hpp`, `src/BlockEncoding.cpp` — block-encoding class and helpers.
- `include/Lcu.hpp`, `src/Lcu.cpp` — LCU decomposition (Pauli strings and coefficients).
- `include/KAKDecomposition.hpp`, `src/KAKDecomposition.cpp` — KAK decomposition for two-qubit unitaries.
- `include/CSDecomposition.hpp`, `src/CSDecomposition.cpp` — cosine-sine decomposition for even-dimensional unitaries.
- `include/Gate.hpp`, `src/Circuit.cpp` — native gate type and dense/simulator circuit evaluation shared by the synthesizers.
- `include/TwoQubitSynthesis.hpp`, `src/TwoQubitSynthesis.cpp` — compile a 2-qubit unitary to native gates (KAK -> single-qubit gates + CNOTs).
- `include/ShannonDecomposition.hpp`, `src/ShannonDecomposition.cpp` — compile an arbitrary n-qubit unitary to native gates (recursive CSD).
- `include/Qsp.hpp`, `src/Qsp.cpp` — QSP helper for small systems.
- `include/QspAngleSolver.hpp`, `src/QspAngleSolver.cpp` — QSP angle finding (phase factors for a target polynomial).
- `include/Gate.hpp`, `src/Circuit.cpp` — also: `countResources` (gate/depth) and `toQasm` (OpenQASM 2.0 export).
- `include/Qsvt.hpp`, `src/Qsvt.cpp` — simulator-owning QSVT engine wrapper.
- `include/QsvtPipeline.hpp`, `src/QsvtPipeline.cpp` — full QSVT (block-encode -> QSVT operator -> circuit) and the `compileMatrixFunction` matrix-function compiler.
- `include/HamiltonianSimulation.hpp`, `src/HamiltonianSimulation.cpp` — `e^{-iHt}` via QSVT (cos(tH) and sin(tH) circuits).
- `include/EigenvalueThreshold.hpp`, `src/EigenvalueThreshold.cpp` — spectral projector / eigenvalue thresholding via QSVT sign(H-mu).
- `tests/test_qsvt.cpp` — GoogleTest suite (decompositions, QSVT block == P(A), QSVT circuit on Qrack, QASM round-trip, resources, block-encoding, LCU).
- `bench/` — benchmark harness comparing this framework against Qiskit and PennyLane; see [BENCHMARK.md](BENCHMARK.md).
- `bindings/` — optional pybind11 Python bindings (`qsvt_native`): NumPy in/out, exposes the angle solver, decompositions (+ QASM/resources), QSVT pipeline, and applications. See [bindings/README.md](bindings/README.md). Build with `-DBUILD_PYTHON=ON`.
- `examples/` — runnable Python demo (`qsvt_demo.py`) and notebook (`qsvt_demo.ipynb`) covering the full pipeline + applications.
- `docs/angle-solver.md` — technical note on the homotopy + analytic-Jacobian QSP angle solver.

Known problems and design decisions
----------------------------------

- Block encoding for n-qubit gates (n >= 2):
  - Status: resolved. The `BlockEncoding` matrix constructor builds and stores a
    valid block-encoding *unitary* (the dilation embedding the operator as the
    top-left block), and `apply(qubits)` compiles it into native gates via the
    Quantum Shannon Decomposition and runs it on hardware-like circuits. What
    remains is gate-count optimisation (next steps).
  - I attempted to implement block encoding for an arbitrary n-qubit operator that acts as the top-left block of a larger unitary. When doing that for multi-qubit matrices I found there was no simple, already-implemented routine in the codebase to directly produce the hardware-style decomposition for n >= 2.
  - Two routes were considered:
    1. Use the statevector / large-unitary approach: build a big unitary matrix (embedding your target as a block) and feed it to the simulator as a single matrix operation. This works in simulation but doesn't reflect how the unitary would be implemented on hardware.
    2. Implement the hardware-like decomposition: decompose the block-encoding into native gates and controlled operations. This is harder but gives correct gate-level construction for later compilation or hardware use. I chose the second, harder route.

- After choosing the hardware-style path, I studied gate decomposition techniques. I investigated Lie theory and Lie algebras because many decompositions (KAK, Cartan/KAK for SU(4)) are derived from Lie-group/Lie-algebra structure. While the math is heavy, the practical takeaway is that special decompositions let you express two-qubit unitaries as local gates + a canonical entangling piece.

- I also considered the Linear Combination of Unitaries (LCU) algorithm as another route to implement more general operations. LCU can be used to implement operators expressed as weighted sums of unitaries, but it requires ancilla preparation and controlled-selection mechanisms; the same practical issue (how to implement the controlled selection efficiently) arises.

- To address the decomposition problem, I focused on implementing KAK decomposition (for 2-qubit unitaries) and the Cosine–Sine (CS) decomposition (for higher-dimensional block decompositions). Both of these decompositions are related to Lie-group decompositions and matrix factorization techniques — they use similar mathematical machinery (e.g., diagonalization of certain associated matrices, Cartan decomposition ideas) to produce sequences of simpler gates. Note: KAK is a Cartan decomposition specialized to SU(4); CS decomposition is a matrix factorization that is often used to split a unitary into simpler blocks and is commonly used in multi-qubit decomposition.

Why KAK and CS (briefly and simply)
-----------------------------------

- KAK decomposition: expresses a two-qubit unitary U as U = K1 * A * K2 where K1 and K2 are local (single-qubit) operations and A is a canonical entangling operation parameterized by three angles. This is very useful because once A is known, you can implement the whole two-qubit gate with a small number of CNOTs and single-qubit rotations.
- CS decomposition: splits a unitary into blocks using cosine-sine matrices and is useful when decomposing multi-qubit gates into smaller units.

QSP and the angle-finding challenge
-----------------------------------

Quantum Signal Processing (QSP) is the key algorithmic building block that allows the implementation of polynomial transformations of eigenvalues (or singular values when combined with block-encoding). Practically, QSP requires a sequence of single-qubit rotations whose angles are chosen so that the resulting unitary implements the desired polynomial on a target subspace.

The hard part is the angle-finding algorithm: given a target polynomial (or filter) you need to compute the sequence of phases (angles) that yield that polynomial. There are established algorithms described in the literature (Low & Chuang and subsequent works) that compute these phases via root-finding and leveraging properties of Chebyshev polynomials and phase factorization. Implementing a numerically stable and robust solver for those phases is one of the core challenges of this project.

Where QSP is useful (examples)
- Hamiltonian simulation (approximate e^{-iHt} by polynomials of H or block-encoded H)
- Singular-value transformation and matrix functions (apply f(Σ) in SVD)
- Quantum linear system algorithms (HHL-style and improvements using QSVT/LCU)
- Amplitude amplification and fixed-point amplitude amplification
- Quantum machine learning primitives that use polynomial approximations to kernels or activation functions

Planned next steps (concrete)
-----------------------------

DONE: the full QSVT pipeline is implemented (`QsvtPipeline`) and the pieces are
integrated into a matrix-function compiler with a matrix-inversion example.
`block(U_Phi) == P(A)` is verified to machine precision and the circuit runs on
the Qrack simulator.

1. `QspAngleSolver` — largely addressed via homotopy continuation (now solves to
   machine precision at degree 25+ and near `|f| = 1`; a degree-25 regularized
   inverse drives the matrix-inversion demo to ~2% on the spectrum). For very
   high degree / arbitrary precision, the next step is the
   complementary-polynomial completion + root-finding / Fejer-Riesz method (as
   in pyqsp), which is non-iterative and machine-precision by construction.
2. Further reduce CNOT count toward the optimal ~0.48*4^n. Done so far: Mottonen
   uniformly-controlled rotations (2^k CNOTs) + 4-CNOT 2-qubit base + peephole
   (~0.58*4^n, 1.4x above optimal). Remaining: the optimal 3-CNOT 2-qubit base
   (Vatan-Williams Weyl-template fit; ~1.25x) and the Shende-Bullock-Markov
   cross-level merges.
3. More QSVT applications on top of the compiler. Done: matrix inversion,
   Hamiltonian simulation (`e^{-iHt}`), and eigenvalue thresholding / spectral
   projection (`EigenvalueThreshold`). Next: amplitude amplification, and
   combining multi-part circuits (e.g. cos/sin of Hamiltonian sim) into a single
   circuit via LCU (one extra ancilla).
4. Implement full LCU operator application and controlled-selection primitives
   (PREPARE/SELECT). General (non-Hermitian) singular-value QSVT (two projectors).

Build and run instructions
--------------------------

This project uses CMake. The top-level `CMakeLists.txt` provides options to set Qrack paths and to toggle CUDA support.

Simple out-of-source build (no Qrack installed / tests off):

```bash
mkdir -p build
cmake -S . -B build -DBUILD_TESTS=OFF -DUSE_CUDA=OFF
cmake --build build -j$(nproc)
```

If Qrack is installed on your machine, point CMake to its headers and library (adjust paths):

```bash
cmake -S . -B build \
  -DQRACK_INCLUDE_DIRS=/usr/local/include/qrack \
  -DQRACK_LIBRARIES=/usr/local/lib/libqrack.so \
  -DBUILD_TESTS=ON -DUSE_CUDA=OFF
cmake --build build -j$(nproc)
```

If Qrack provides a CMake config (QrackConfig.cmake), prefer adding its install prefix to CMAKE_PREFIX_PATH:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/qrack/install
cmake --build build -j
```

If you enable CUDA support, ensure Qrack was built with CUDA and the machine has CUDA drivers installed:

```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_PREFIX_PATH=/path/to/qrack/install
cmake --build build -j
```

If tests fail to link with undefined references to Qrack symbols, it means CMake did not find or link the `libqrack` library. Re-run CMake with `QRACK_INCLUDE_DIRS` and `QRACK_LIBRARIES` set to the correct locations or set `CMAKE_PREFIX_PATH` to the Qrack install prefix.

Notes and help
--------------

- The KAK and CS decomposition code lives in `src/` and `include/`. If you are experimenting with decompositions, the KAK implementation is a good place to begin because it focuses on two-qubit canonical decomposition.
- The `BlockEncoding` class currently contains a simple scalar constructor and some matrix checks; the full multi-qubit block construction (hardware-style) is still under development.
- If you want me to add reference links (papers, lecture notes) for KAK, CS decomposition, LCU, and QSP I can add a `docs/` section with canonical references (Low & Chuang, Nielsen & Chuang, etc.).

Contact / contribution
----------------------

This is an ongoing personal project to explore QSVT internals and gate-level implementations. Contributions, PRs, and issue reports are welcome — especially test cases for decompositions, numeric solvers for QSP phasing, and small reproducible examples for block-encoding.

Acknowledgements
----------------

- Uses Qrack for the quantum simulator backend ([https://github.com/vmorgner/qrack](https://github.com/unitaryfoundation/qrack))
- Eigen for linear algebra

---

If you want, I can also:

- Add a short `docs/` folder with references and math notes for KAK and CS decomposition.
- Add example notebooks (Python/C++) that show how to run small end-to-end examples.
- Produce a short tutorial section that walks through a simple 1- or 2-qubit QSP example and the corresponding angle calculations.

Tell me which of those you'd like next and I will prepare it.

Selected research papers and resources
-------------------------------------

Below are curated papers and resources that are highly relevant to the topics in this repository (QSP, QSVT, block-encoding, LCU, KAK decomposition, and CSD). They are good starting points for deepening the theoretical background and for implementing robust numerical routines (angle-finding, decomposition algorithms, etc.).

- Gilyén, A., Su, Y., Low, G. H., & Wiebe, N. — "Quantum singular value transformation and beyond" (2019). Introduces the QSVT framework and the block-encoding formalism; this is the central modern reference for QSVT. https://arxiv.org/abs/1806.01838

- Low, G. H., & Chuang, I. L. — "Hamiltonian simulation by qubitization" (2019). Presents qubitization and QSP-based techniques for optimal Hamiltonian simulation. (See also Low & Chuang's related work on Quantum Signal Processing). https://arxiv.org/abs/1805.00675

- Low, G. H., & Chuang, I. L. — "Quantum signal processing" (foundational materials / lecture notes). These works describe the single-qubit signal-processing primitives and how phase sequences implement polynomial transforms.

- Berry, D. W., Childs, A. M., Cleve, R., Kothari, R., & Somma, R. D. — "Simulating Hamiltonian Dynamics with a Truncated Taylor Series" (2015). Introduces LCU-related techniques applied to Hamiltonian simulation. https://arxiv.org/abs/1501.01715

- Childs, A. M., & Wiebe, N. — "Hamiltonian simulation using linear combinations of unitary operations" (2012). One of the early works describing the LCU technique in detail.

- Vatan, F., & Williams, C. — "Optimal quantum circuits for general two-qubit gates" (2004). A practical reference on optimal two-qubit gate decompositions; closely related to KAK-style optimizations. https://arxiv.org/abs/quant-ph/0308006

- Khaneja, N., Brockett, R., & Glaser, S. J. — "Time optimal control in spin systems" and related works on Cartan/KAK decompositions in quantum control. These papers discuss Cartan decompositions and control-theoretic views on unitary factorization.

- Möttönen, M., Vartiainen, J. J., Bergholm, V., & Salomaa, M. M. — "Transformation of quantum states using uniformly controlled rotations" (2004). Shows constructions using uniformly controlled rotations; related to multiplexed gates and CSD-based decompositions. https://arxiv.org/abs/quant-ph/0407010

- Shende, V. V., Bullock, S. S., & Markov, I. L. — "Synthesis of quantum logic circuits" (2004/2006). Discusses general unitary decomposition strategies and practical circuit synthesis including CSD approaches.

- Golub, G. H., & Van Loan, C. F. — "Matrix Computations" (textbook). Contains background on the Cosine–Sine decomposition (CSD) and practical numerical methods for matrix factorizations.

- Nielsen, M. A., & Chuang, I. L. — "Quantum Computation and Quantum Information" (book). Standard textbook with background on quantum gates, decompositions, and algebraic structure.

Additional resources and implementations
--------------------------------------

- QSVT & QSP lecture notes and tutorials by the community (search for "Quantum Signal Processing lecture notes" and "Quantum singular value transformation tutorial"). These are often easier to digest than formal papers when implementing algorithms.
- Numerical tools for phase/angle finding: several community implementations exist (MATLAB/NumPy/Julia), and you can often find code accompanying papers by Low & Chuang or Gilyén et al.

If you'd like, I can add a `docs/` page with direct links to these papers and short notes about which sections to read first for implementation (e.g., which parts of Gilyén et al. cover block-encoding; which parts of Low & Chuang detail angle synthesis). I can also add citations in BibTeX format if you prefer.
