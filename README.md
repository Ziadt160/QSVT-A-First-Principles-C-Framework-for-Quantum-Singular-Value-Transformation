# QSVT — A First-Principles C++ Framework for Quantum Singular Value Transformation

Short project summary
---------------------

This repository is a work-in-progress C++ framework that explores and implements the pieces required to build Quantum Singular Value Transformation (QSVT) from the ground up. The code uses Qrack as the quantum simulator backend (Qrack can use CUDA to accelerate simulation). The goal is to implement the low-level building blocks (block-encoding, LCU, gate decompositions, QSP) and then integrate them into a full QSVT pipeline.

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

- BlockEncoding: basic scalar block-encoding implemented; matrix constructor starts checks but the large-unitary construction is incomplete (see known issues).
- Lcu: generation of Pauli string basis and coefficients implemented for representing matrices as linear combinations of Pauli operators.
- Qsp: a one-qubit QSP routine and a small driver that applies a sequence of rotations / unitaries using a given angle set.
- KAKDecomposition and CS/CSD: work is in progress — KAK decomposition implementation exists for 2-qubit unitaries (SU(4)) and returns K1, A, K2 factors.
- QSVT: a thin wrapper around the Qrack interface that will orchestrate the full pipeline (simulator creation, ancilla bookkeeping).

Files of interest
-----------------

- `include/BlockEncoding.hpp`, `src/BlockEncoding.cpp` — block-encoding class and helpers.
- `include/Lcu.hpp`, `src/Lcu.cpp` — LCU decomposition (Pauli strings and coefficients).
- `include/KAKDecomposition.hpp`, `src/KAKDecomposition.cpp` — KAK decomposition for two-qubit unitaries.
- `include/Qsp.hpp`, `src/Qsp.cpp` — QSP helper for small systems.
- `include/Qsvt.hpp`, `src/Qsvt.cpp` — top-level QSVT wrapper.
- `tests/test_qsvt.cpp` — test harness and small examples.

Known problems and design decisions
----------------------------------

- Block encoding for n-qubit gates (n >= 2):
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

1. Finish robustly implementing KAK and CS/CSD decompositions and add test cases. These are necessary to realize hardware-style block encodings for multi-qubit operators.
2. Finish the BlockEncoding class to support n-qubit matrices with a hardware-like decomposition (not just the statevector embedding).
3. Implement full LCU operator application and controlled-selection primitives needed for LCU-based implementations.
4. Implement the QSP angle-finding routine (numerical solver for QSP phases) and add utilities to synthesize QSP circuits for multi-qubit block-encodings.
5. Integrate the pieces to implement QSVT and test on canonical examples (matrix inversion, Hamiltonian simulation, singular value thresholding).

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
