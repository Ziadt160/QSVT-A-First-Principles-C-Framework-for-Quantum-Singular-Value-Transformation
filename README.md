# QSVT — A First-Principles C++ Framework for Quantum Singular Value Transformation

![tests](https://img.shields.io/badge/tests-38%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![lang](https://img.shields.io/badge/C%2B%2B-17-00599C)

A from-scratch C++ implementation of Quantum Singular Value Transformation that
compiles a matrix function `f(A)` into a runnable quantum circuit — block-encoding
→ QSP angle finding → native-gate synthesis → OpenQASM and resource estimates —
entirely in compiled C++, with a Qrack GPU-simulator backend and optional Python
bindings.

**Its phase-angle solver is, as far as an exhaustive search can establish, the only
QSP angle solver written in a compiled language that is usable today.** Every
actively maintained alternative — `pyqsp`, `QSPPACK`, `nlft-qsp`, PennyLane — is
Python or MATLAB. The one other compiled implementation, Microsoft's F# port, has
been archived since January 2024 under a non-commercial licence. The core here is
**stdlib-only C++17** — six standard headers and nothing else. That is a CI job,
not a claim: every push compiles the solver and its C ABI in an empty directory
with **no `-I` flags at all** under `-Werror`, then links them from pure C. A
`pip install` needs no Eigen and no network; `cargo build` needs only a compiler.

| | |
|---|---|
| **Angle finding** | **1.4×–17× faster** than the best external solver across six (κ, slack) configurations, at machine-precision residual |
| **Two algorithms** | symmetric-QSP Newton **and** inverse NLFT, both implemented in C++ and validated against their reference implementations |
| **Target compilation** | function + domain + accuracy → validated angles, **or an actionable reason none exist** — a step no other package performs |
| **Block encoding** | native-gate LCU, **5.5× fewer CNOTs than Qiskit's** Quantum Shannon Decomposition at n=5 |
| **Applications** | H₂ and Fermi–Hubbard ground states; quantum linear systems; a measured (κ, ε) resource map; the regime where transform-based methods stop working entirely |
| **Verification** | 38/38 tests, cross-validated against `pyqsp` to ~15 digits, reproducible pinned benchmarks |

Results at a glance
-------------------

- **Angle finding — [`qsp-angles`](packages/qsp-angles).** A C++ implementation of
  the symmetric-QSP Newton method (Dong–Lin–Ni–Wang, arXiv:2307.12468),
  cross-validated against `pyqsp` to ~15 digits and reaching degree 1000+ at
  machine precision. On the QSVT `1/x` inversion target it is **1.4×–17× faster
  than the best external solver across six (κ, slack) configurations** — measured
  against `pyqsp`, `QSPPACK` and `nlft-qsp`, single-threaded, pinned, median of 3
  ([BENCHMARK.md](packages/qsp-angles/bench/BENCHMARK.md)). At degree 1001:
  **3.6 s against 10.2, 40.7 and 70.1**.
- **A second algorithm, for the opposite regime.** Weiss completion plus a
  Half-Cholesky inverse nonlinear Fourier transform — also written in C++ and
  validated against the reference implementation to ~1e-12. The two solvers have
  *opposite* cost profiles: Newton is flat in the target's slack η = 1 − sup|p|,
  NLFT scales as O(1/η). At degree 505, Newton runs 5.43 s → 5.07 s across η while
  NLFT runs 22.4 s → 0.27 s. The crossover is measured at η ≈ 3e-3. Implementing
  and characterising both is what makes the choice a measurement rather than a
  guess ([NLFT_PORT_PLAN.md](packages/qsp-angles/docs/NLFT_PORT_PLAN.md)).
- **A QSP *target* compiler — the step every other package skips.** They all take
  polynomial *coefficients* and trust that you produced a valid target. That is
  where things actually break: uniform-grid fits silently yield |p| > 1 above
  degree ~600, `erf` overshoots the feasibility boundary under truncation, and a
  hand-picked degree too low for its κ simply cannot converge — each of which
  looks like a solver failure and is not.
  [`qsp_compile`](packages/qsp-angles/qsp_angles/compile.py) takes a **function**,
  a domain and an accuracy; selects the minimal degree; fits on
  Chebyshev-clustered nodes; validates the sup-norm; escalates shave, then degree;
  and returns validated angles **or an actionable reason none exist**. Verified
  against every failure mode above, including two that it correctly *refuses*.
- **Sparse block-encoding — avoids dense `4ⁿ` synthesis:** a native-gate LCU
  (PREPARE+SELECT) block-encoding of a local Hamiltonian, validated exact
  (`block = A/α` to ~1e-14). Against **Qiskit's** Quantum Shannon Decomposition
  on the same dilation it is **5.5× fewer CNOTs at n=5** and growing with n.
  (A larger ~24× figure appears in some notes — that compares against *our own*
  dense `0.58·4ⁿ` formula, not a real competitor, and ~1.5× of it is
  subnormalization the caller repays in polynomial degree, since the LCU encodes
  `A/α` while the dense dilation encodes `A/‖A‖`.) See the honest
  [scaling/GPU analysis](applications/qls/README.md).
- **Full QSVT pipeline, validated end-to-end** (dense, to ~1e-16): local
  Hamiltonian → LCU block-encoding → qubitization walk operator → QSVT with
  `sym_qsp` angles → the target polynomial applied to `A/α`.
- **Applications on REAL data** ([`applications/`](applications)):
  - **Quantum linear systems** — solves `A x = b` with state **infidelity
    7e-10 – 2e-7** vs exact (up to 16×16, condition numbers 4–16); the operator
    error is ~5e-4. (Earlier notes said "fidelity 1.0"; that was `%.6f`
    rounding, and fidelity is quadratically insensitive here — `1 − F` is the
    honest metric.)
  - **Ground-state energy** of the **H₂ molecule** (STO-3G; O'Malley et al. 2016)
    and the **Fermi–Hubbard** material model, end to end. The QSVT pipeline
    reproduces exact diagonalization of the given Hamiltonian to ~0 mHa; the
    often-quoted 0.036 mHa is the gap between two *published literature
    constants*, and is a property of the input data, not of this code.
  - **Where transform methods stop working** — spectral/DST kernels are *exact*
    on the constant-coefficient heat equation because the sine basis diagonalises
    it. Make the medium layered
    ([`variable_coefficient.py`](applications/qls/variable_coefficient.py)) and the
    sine basis stops diagonalising (off-diagonal mass 6e-15 → 0.46): the spectral
    solve lands at **38% error** while QSVT holds ~3e-4. Add advection
    ([`advection_diffusion.py`](applications/qls/advection_diffusion.py)) and the
    operator is non-symmetric, so its diagonalising transform is not unitary and
    there is no circuit for it at all — while QSVT holds ~4e-4 by raising degree
    19 → 221. QSVT wins here on **applicability**, not efficiency.
  - **A measured resource map** over 28 (κ, ε) configurations, degrees 9–1985, in
    468 s ([`resource_phase_diagram.py`](applications/qls/resource_phase_diagram.py)):
    degree scales as **κ^1.03** (the optimal rate) but end-to-end cost as
    **κ^3.05** per accepted sample (κ^2.04 with amplitude amplification) — the
    factor asymptotic tables tend to omit.
- **Gate synthesis** (random unitary → native gates), verified vs Qiskit's QSD:
  ~1e-13 reconstruction, **4 / 28 / 136 / 592 CNOTs** for n = 2..5 (~1.4× optimal).
- **Verification**: GoogleTest suite (dense + Qrack simulator), independent
  cross-validation vs `pyqsp`, reproducible pinned benchmarks (CPU + CUDA GPU),
  OpenQASM export, and CI.

What this does *not* claim
--------------------------

Every number above is reproducible from a script in this repository. These are the
limits of what they show — stated here rather than buried, because a benchmark
without its boundary conditions is marketing.

- **The speed margin over `pyqsp` and `QSPPACK` is a compiled-vs-interpreted
  constant** on the same Newton algorithm — not a better algorithm.
- **`nlft-qsp` overtakes this solver between degree 1001 and 1501** on hard
  targets, and is 3.8× faster by degree 4001. Its inverse-nonlinear-FFT scales as
  O(d log² d) against this Newton core's ~O(d^2.8). Above ~1500, it is the right
  tool — which is precisely why the second algorithm is implemented here too.
- **Automatic dispatch between the two solvers is not yet wired.** The shipped C
  ABI is the Newton core; the NLFT path is available but not yet selected for you.
- **The applications are validated at the operator level** (dense linear algebra),
  as is standard for this class of result and as the comparison points also do.
  Sparse circuit execution on Qrack at larger *n* is the remaining step.
- **QSVT is not free.** The measured resource map puts end-to-end cost at
  **κ^3.05 queries per accepted sample** (κ^2.04 with amplitude amplification).
  Where it beats transform-based methods, it wins on *applicability*, not
  efficiency.
- **Angles cache.** For a well-conditioned parabolic PDE they are computed once —
  measured: 3 distinct parameter sets across 118 adaptive timesteps. The case for
  a fast embeddable solver is parameter studies and deployment without an
  interpreter, not per-step regeneration. That negative result is committed too,
  in [`adaptive_stepping.py`](applications/qls/adaptive_stepping.py).

```bash
cmake -S . -B build && cmake --build build -j      # builds (Release by default)
ctest --test-dir build                              # 38 tests
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

Status & roadmap
----------------

The full QSVT pipeline is implemented and verified: `block(U_Phi) == P(A)` to
machine precision, running on the Qrack simulator. Since the original write-up,
most of the roadmap is done — see [`packages/qsp-angles`](packages/qsp-angles) and
[`applications/`](applications):

**Done:**
1. **Angle finding** — a C++ port of the symmetric-QSP Newton method (`sym_qsp`):
   machine precision to **degree 1000+**, **14–69× faster** than pyqsp depending
   on degree (the margin narrows as degree grows), cross-validated against it,
   and extracted as an **embeddable** package. (The Fejér–Riesz direction is no
   longer needed for this regime.)
2. **Sparse block-encoding** — full LCU **PREPARE/SELECT** native-gate primitives
   (`LcuBlockEncoding`), validated exact, **5.5× fewer CNOTs at n=5** than
   Qiskit's Quantum Shannon Decomposition on the same dilation, growing with n.
3. **Applications** — quantum linear systems (state infidelity 7e-10 – 2e-7), and
   ground-state energy on **real data** (H₂ molecule and Fermi–Hubbard, matching
   exact diagonalization of the given Hamiltonian). The whole sparse pipeline
   (LCU → qubitization walk → QSVT with `sym_qsp` angles) is validated end to end.

**Remaining:**
- Wire `LcuBlockEncoding` into the C++ QSVT run and execute on Qrack at large n
  (the dense math is already validated); harden the circuit-on-Qrack path.
- Optimal 3-CNOT 2-qubit base + Shende–Bullock–Markov merges (dense synthesis);
  T-count resource estimates; amplitude amplification; the H₂ dissociation curve
  and LiH; general (non-Hermitian) singular-value QSVT (two projectors).

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

- The KAK and CS decomposition code lives in `src/` and `include/`. The KAK
  implementation is a good place to start (two-qubit canonical decomposition).
- Block-encoding: the dense multi-qubit dilation (`rotationBlockEncoding`,
  compiled via the Quantum Shannon Decomposition) and the sparse LCU
  block-encoding (`LcuBlockEncoding`, poly(n) gates) are both implemented and
  validated. See [`applications/qls`](applications/qls) for the sparse path.
- Canonical references (QSVT/QSP/block-encoding/LCU/KAK) are listed at the bottom
  of this README.

Contact / contribution
----------------------

This is an ongoing personal project to explore QSVT internals and gate-level implementations. Contributions, PRs, and issue reports are welcome — especially test cases for decompositions, numeric solvers for QSP phasing, and small reproducible examples for block-encoding.

Acknowledgements
----------------

- Uses Qrack for the quantum simulator backend ([https://github.com/vmorgner/qrack](https://github.com/unitaryfoundation/qrack))
- Eigen for linear algebra

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
- Numerical tools for phase/angle finding: `pyqsp` (Python) and `QSPPACK`
  (MATLAB); this project's `sym_qsp` is a validated, faster, embeddable C++
  implementation of the same method (see [`packages/qsp-angles`](packages/qsp-angles)).

Citing this work
----------------

If you use this in academic work, please cite it. Machine-readable metadata lives
in [CITATION.cff](CITATION.cff) — GitHub renders a "Cite this repository" button
from it, and Zenodo reads it when minting a DOI for a release.

<!-- Once the first GitHub release is archived by Zenodo, paste the CONCEPT DOI
     (the version-independent one, which always resolves to the newest release)
     here and as a badge at the top of this file. -->

    Mohammed, Z. T. (2026). QSVT: A First-Principles C++ Framework for Quantum
    Singular Value Transformation (Version 0.1.0) [Computer software].
    https://github.com/Ziadt160/QSVT-A-First-Principles-C-Framework-for-Quantum-Singular-Value-Transformation

The two solver algorithms are due to their original authors; if you use them,
please cite those papers too — Dong, Lin, Ni & Wang (arXiv:2307.12468) for the
symmetric-QSP Newton method, and Ni & Ying (arXiv:2410.06409) for the
inverse-NLFT approach. Both are listed in `CITATION.cff`.
