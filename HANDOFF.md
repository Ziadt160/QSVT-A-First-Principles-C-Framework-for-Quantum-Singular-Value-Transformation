# Project handoff / continuation guide

A snapshot of state, environment, and open work so a new session can continue
without re-deriving context. (Project overview + API live in [README.md](README.md).)

## 1. What this is

A from-scratch, verified C++ implementation of Quantum Singular Value
Transformation. It compiles a matrix function `f(A)` into a runnable quantum
circuit: block-encoding -> QSP angle finding -> KAK/CS/Quantum-Shannon native-gate
synthesis -> OpenQASM + resource estimates. Backend: Qrack (GPU statevector).
Optional Python bindings. Three applications: matrix inversion, Hamiltonian
simulation `e^{-iHt}`, eigenvalue thresholding. **29 GoogleTest cases pass**
(dense + Qrack-simulator verified).

## 2. Components (all under `namespace qsvt`)

| Area | Files |
|------|-------|
| Shared types | `Common.hpp`, `QrackTypes.hpp` |
| Block-encoding | `BlockEncoding.{hpp,cpp}` |
| LCU (Pauli decomposition) | `Lcu.{hpp,cpp}` |
| KAK (2-qubit Cartan) | `KAKDecomposition.{hpp,cpp}` |
| Cosine-Sine decomposition | `CSDecomposition.{hpp,cpp}` |
| 2-qubit synthesis (4-CNOT) | `TwoQubitSynthesis.{hpp,cpp}` |
| Quantum Shannon Decomp. | `ShannonDecomposition.{hpp,cpp}` |
| Gate type, dense/sim circuit, resources, QASM | `Gate.hpp`, `Circuit.cpp` |
| QSP angle solver (homotopy + analytic Jacobian) | `QspAngleSolver.{hpp,cpp}` |
| Full QSVT + matrix-function compiler | `QsvtPipeline.{hpp,cpp}` |
| Hamiltonian simulation | `HamiltonianSimulation.{hpp,cpp}` |
| Eigenvalue thresholding | `EigenvalueThreshold.{hpp,cpp}` |
| QSVT engine (simulator owner) | `Qsvt.{hpp,cpp}` |
| Tests | `tests/test_qsvt.cpp` (29 cases) |
| Benchmark vs Qiskit/PennyLane | `bench/` (`qsvt_bench` + `benchmark.py`) |
| Python bindings | `bindings/` (`qsvt_native`, pybind11) |
| Examples | `examples/qsvt_demo.{py,ipynb}` |
| Docs | `docs/angle-solver.md`, this file |

## 3. Git / CI state (IMPORTANT)

- Repo: `github.com/Ziadt160/QSVT-A-First-Principles-C-Framework-for-Quantum-Singular-Value-Transformation`
- Working branch: `claude/nervous-spence-3f139a`.
- **`main` is currently RED.** PR #2 merged the framework + polish + docs, but it
  contained a QASM angle-extraction bug (now fixed).
- **The fix is commit `b80431e`, pushed to the branch but NOT yet merged to
  `main`.** ACTION: open a new PR for the branch (1 commit ahead of `main`) and
  merge it; CI will then go green. PR link:
  `.../pull/new/claude/nervous-spence-3f139a`
- After green, swap the static `tests` badge in the README for the live CI badge.
- CI workflow: `.github/workflows/ci.yml` — installs Eigen (apt), **builds Qrack
  from source CPU-only**, builds the project, runs `ctest`. Verified locally
  against the same CPU Qrack (v10.10.2): 29/29.

## 4. Build / test / run

```bash
# C++ (default Release; -march=native is opt-in via -DQSVT_NATIVE_ARCH=ON)
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_BENCH=ON -DUSE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure   # 29 tests
./build/src/qsvt_app                          # demo

# Benchmark vs Qiskit/PennyLane
./build/bench/qsvt_bench /tmp/bench_input.json
python3 bench/benchmark.py /tmp/bench_input.json

# Python bindings (needs python3-dev + `pip install pybind11`)
cmake -S . -B build -DBUILD_PYTHON=ON && cmake --build build --target qsvt_native
```

## 5. Environment gotchas (this dev box)

- **Git must run from the WINDOWS side** (the Bash tool / PowerShell), not WSL:
  the worktree's `.git` gitdir is a `//wsl.localhost/...` UNC path that WSL git
  can't resolve (`fatal: not a git repository`). One-time:
  `git config --global --add safe.directory '%(prefix)///wsl.localhost/.../nervous-spence-3f139a'`.
- **Builds run in WSL** (`wsl.exe -- bash -lc '...'`): g++ 12.4, cmake, Eigen at
  `/usr/include/eigen3`, Qrack at `/usr/local` (float/GPU build), CUDA present.
  The Windows-side Bash tool has msys2 g++ but no Eigen/Qrack — use WSL to build.
- WSL repo path: `/home/ziad/QSVT_Project/.claude/worktrees/nervous-spence-3f139a`.
- **No sudo, no `python3-dev`, no `gh`.** Workarounds used:
  - pip bootstrapped via `get-pip.py` to `~/.local`; `numpy scipy qiskit pennylane
    pybind11` installed with `pip install --user --break-system-packages`.
  - Python dev headers fetched without root: `apt-get download libpython3.12-dev
    python3.12-dev` then `dpkg -x ... /tmp/pydev/extracted` (note: `/tmp` is wiped
    between sessions — re-fetch as needed).
  - Python extension compiled manually (no system python3-dev for the CMake path);
    exact command in `bindings/README.md`. An extension `.so` does NOT link
    libpython.
  - CI logs require auth and `gh` is absent, so CI failures were diagnosed by
    reproducing the environment locally (building Qrack CPU-only).
- Newer Qrack removed `qrack/hamiltonian.hpp`; the CMake Qrack-finder keys on it.
  Not a problem with the installed Qrack, but note if upgrading.

## 6. Bugs found and fixed this session (context for trust)

- **KAK** produced non-local `K1` for ~40% of inputs: it landed in O(4)\SO(4)
  (det -1); fixed with a sqrt-branch parity correction. Also switched to proper
  simultaneous diagonalization of `Re(M)`/`Im(M)` with degenerate-subspace handling.
- **QSP solver**: `Phi=0` is a stationary point (exact gradient zero) -> start the
  homotopy slightly off it. Finite-diff hid this; the analytic Jacobian exposed it.
- **QASM U3 extraction** averaged `phi+lambda` and `phi-lambda` (each mod 2*pi) ->
  inconsistent wrapping flipped off-diagonal signs. Now extract `phi`, `lambda`
  directly relative to `arg(M00)`. (This was the CI failure.)
- **Build was `-O0`** (no `CMAKE_BUILD_TYPE`) -> default Release; `JacobiSVD ->
  BDCSVD`; `-march=native` made opt-in (portability + CI codegen).

## 7. Benchmark headline (vs Qiskit/PennyLane, identical inputs)

- Decomposition: reproduces unitary to ~1e-13; **4/28/136/592 CNOTs** (n=2..5),
  ~1.4x Qiskit's optimal QSD; competitive/faster per-call.
- QSP angle solver: machine precision to **degree 100+** (degree 101 ~1e-14 in
  ~1.7 s); near `|f|=1` too.
- Applications: inversion ~2% on spectrum (deg 25); `e^{-iHt}` ~1e-13; projector
  ~5e-3. Full table in [BENCHMARK.md](BENCHMARK.md).

## 8. Roadmap (highest-leverage first)

1. **Merge `b80431e` -> `main`; confirm CI green; swap badge.** (Only open item to
   call the publish loop done.)
2. **Fejer-Riesz / Newton angle solver to degree ~1000** — the one capability that
   makes "first high-degree C++ QSP solver" unambiguous (current homotopy LM is
   reliable to ~degree 100, slower beyond). See `docs/angle-solver.md` sec. 5.
3. **Optimal 3-CNOT 2-qubit base** (Vatan-Williams Weyl-template fit) -> ~1.25x
   Qiskit (currently 4-CNOT/1.4x). Deferred earlier as high-risk; needs the
   template + local-gate recovery. (Task was tracked as "3-CNOT optimal base".)
4. **QASM3 export** (currently QASM2); **LCU-combined single-circuit `e^{-iHt}`**
   (cos/sin currently combined classically); general non-Hermitian SVD-QSVT.

## 9. Leverage (non-code)

Strongest path is **portfolio / job signal** (correct nontrivial quantum algorithm
in C++, tests, benchmarks, bindings). Also: contribute the QSVT/angle-finding
layer to **Qrack / Unitary Foundation** (the backend) for a real user base; a
short technical note/blog from `docs/angle-solver.md`. It is a niche-within-a-niche
(QSVT is fault-tolerant-era; circuits don't run usefully on 2026 hardware) — value
today is compilation, resource estimation, verification, teaching. Do not pitch it
as a Qiskit/PennyLane replacement.
