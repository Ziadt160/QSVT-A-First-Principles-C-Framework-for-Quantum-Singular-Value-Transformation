# qsp-angles

**An embeddable C++ QSP/QSVT angle (phase-factor) solver — a state-of-the-art
symmetric-QSP Newton method you can `#include` into a compiled quantum toolchain,
with no Python at the core. Validated against `pyqsp` to machine precision,
reaching degree 1000+.**

Quantum Signal Processing and Quantum Singular Value Transformation both reduce
to one classical pre-computation: given a target polynomial `f(x)`, find the
phase sequence `Φ` that makes a QSP circuit realise `f`. Established solvers
(`pyqsp` in Python, `QSPPACK` in MATLAB) are excellent but interpreted — you
cannot link them into a C++/Rust compiler or simulator without dragging in a
runtime. **This package is the missing embeddable piece: the default `sym_qsp`
core is dependency-free C++ (standard library only — a hand-rolled Bluestein FFT,
no Eigen), callable directly from compiled code.** The pip wheel is a convenience
on top, not the substrate. (The optional `homotopy` fallback solver uses Eigen.)

The default solver is a faithful C++ port of the **symmetric-QSP Newton method**
(`sym_qsp`; Dong–Lin–Ni–Wang, arXiv:2307.12468 — the *same* algorithm `pyqsp`
uses), validated against `pyqsp` to ~15 digits. It converges in ~5 Newton
iterations at any degree and reaches **degree > 1000** at machine precision. It
is not a new or better algorithm — it is the established one, compiled. Because
it runs the identical method, it is faster than `pyqsp` only by the expected
compiled-vs-interpreted constant factor (tens-fold; 13.7–68.7× depending on degree, on the machine in
[bench/BENCHMARK.md](bench/BENCHMARK.md)) — a real but minor point, since
angle-finding is a one-time offline precompute. The reason to reach for this is
**embeddability**, not speed. A homotopy-continuation solver is retained as
`method="homotopy"`.

```bash
pip install .        # not yet on PyPI - install from a checkout
```

```python
import qsp_angles as qa

# From a callable target (|f| <= 1 on [-1,1], parity = degree mod 2):
r = qa.target2angles(lambda x: 0.7 * x, degree=1)
r.phases       # -> numpy array of phase factors (Wx convention)
r.residual     # -> worst-case error vs the target on [-1, 1]
r.converged    # -> bool  (residual < tol; tol defaults to 1e-6)

# From polynomial coefficients — CHEBYSHEV basis by default (QSP/QSVT convention):
qa.poly2angles([0.0, 0.0, 1.0])                  # T_2(x) = 2x^2 - 1
qa.poly2angles([0.0, 0.0, 1.0], basis="monomial")  # x^2

# Evaluate the achieved QSP polynomial:
qa.response(0.3, r.phases)
```

Coefficient inputs are interpreted in the **Chebyshev** basis by default — this
is the standard QSP/QSVT field convention. Pass `basis="monomial"` for ascending
power-basis coefficients. A numpy `Chebyshev` or `Polynomial` object passed
directly is honored in its own basis.

### Circuits (OpenQASM 3 / Qiskit)

Turn solved phases into a runnable single-qubit QSP circuit:

```python
from qsp_angles import circuits

qasm = circuits.qsp_qasm3(r.phases, x=0.3)   # OpenQASM 3 string (no deps)
qc, theta = circuits.to_qiskit(r.phases)     # parameterized QuantumCircuit
                                             # (pip install .[qiskit])
```

Convention (exact, no global phase): `W(x) = rx(-2 arccos x)`, `E(phi) = rz(-2 phi)`.
For full QSVT of a matrix function, use the same phases but replace `W(x)` with
your block-encoding and `E(phi)` with projector-controlled ancilla rotations.

### A familiar, similarly-named convenience wrapper

`QuantumSignalProcessingPhases` is named after the pyqsp entry point so the call
site looks familiar, but it is **not** a drop-in replacement:

```python
from qsp_angles import QuantumSignalProcessingPhases

phases = QuantumSignalProcessingPhases(poly, signal_operator="Wx")
```

It uses the **Wx** convention and raises `RuntimeError` if the solve does not
converge (it discards the full result object, so a silent non-converged return
would be unsafe).

#### Differences from pyqsp

This wrapper is **not interchangeable** with pyqsp — do not expect identical
output:

- **Return shape.** This returns a single numpy ndarray of `d + 1` *full
  symmetric* phases. pyqsp returns a *reduced* phase list, and its `sym_qsp`
  path returns a 3-tuple `(phases, reduced_phases, parity)`.
- **Phase convention.** The phase convention differs; the numeric values are not
  the same even where the call shape matches.
- **Signal operator.** Only `Wx` is supported here.

Use it for convenience when you want just the phase array from this solver — not
as a way to swap `qsp-angles` in behind existing pyqsp code unchanged.

## Why this exists

`pyqsp` (Python) and `QSPPACK` (MATLAB) are the established angle solvers — both
are interpreted and not embeddable in a compiled stack. This package is, to our
knowledge, the **first standalone C++ QSP angle solver**: the default `sym_qsp`
core is dependency-free (standard library only, no Eigen) with no Python in the
loop, so you can call it directly from C++ or any compiled/native pipeline, and
link it without dragging in a runtime. The
pip wheel is a thin pybind11 binding for the Python crowd. It is extracted from a
full from-scratch C++ QSVT framework
([repo](https://github.com/Ziadt160/QSVT-A-First-Principles-C-Framework-for-Quantum-Singular-Value-Transformation)).

Unlike the homotopy solver this package originally shipped, the default
`sym_qsp` method is a like-for-like port of `pyqsp`'s own Newton algorithm, so it
matches `pyqsp` on accuracy (machine precision) and degree reach (1000+). Running
the identical algorithm compiled is also faster — but only by the expected
compiled-vs-interpreted constant factor (tens-fold), and angle-finding is a
one-time offline precompute, so that speed is not the selling point. The selling
point is that you can embed it. See [bench/BENCHMARK.md](bench/BENCHMARK.md) for
the honest, reproducible numbers and methodology.

### Two methods

Select via `method=` on `target2angles` / `poly2angles` (default `"sym_qsp"`):

- **`sym_qsp`** (DEFAULT) — the symmetric-QSP **Newton method** (Dong–Lin–Ni–Wang,
  arXiv:2307.12468), the algorithm behind `pyqsp`'s `sym_qsp`. Converges in ~5
  iterations to machine precision at any degree; reaches degree > 1000.
- **`homotopy`** — the original solver: homotopy continuation (morph the target
  from `T_d`, which `Φ = 0` solves exactly, to `f`, warm-starting each step) plus
  an exact analytic Jacobian (`O(d)` per node via prefix/suffix products). Kept as
  a fallback; competitive only up to ~degree 100.

Indicative timings (Intel i7-9750H, `-O3 -march=native`; see BENCHMARK.md for
provenance — absolute numbers are hardware-dependent):

| method     | target        | degree | residual | time   |
|------------|---------------|--------|----------|--------|
| `sym_qsp`  | `0.8 T_101`   | 101    | ~7e-14   | ~4 ms  |
| `sym_qsp`  | `0.8 T_501`   | 501    | ~8e-12   | ~285 ms|
| `sym_qsp`  | `0.8 T_1001`  | 1001   | ~9e-12   | ~1.0 s |
| `homotopy` | `0.7 T_101`   | 101    | ~1e-14   | ~1.7 s |

## Honest limits

- `sym_qsp` reaches degree **> 1000** at machine precision in ~5 Newton
  iterations — the old "tops out around degree 100" limit applied to the
  `homotopy` method and is **gone** for the default solver.
- The `homotopy` fallback still tops out around degree 100 and is slower at high
  degree; prefer `sym_qsp` (the default) unless you specifically need it.
- Supports the `Wx` convention only.

## Building from source

```bash
pip install .            # needs only a C++17 compiler
pip install -e .[test]   # editable + test deps
pytest tests
```

**The default build has no third-party C++ dependency**: no Eigen, no
`find_package`, no `FetchContent`, no network. It builds offline on a bare
machine with nothing but a compiler. The wheel vendors a handful of C++ files
plus a thin pybind11 layer — no Qrack, no CUDA.

This is enforced, not asserted. The `abi-no-third-party-deps` CI job copies the
solver and the C ABI into an empty directory, compiles them with **no `-I` flags
at all** under `-Werror`, and links the result from pure C. The reachable include
set is six standard headers (`<algorithm> <array> <cmath> <complex> <functional>
<vector>`) and nothing else.

### Optional: the homotopy fallback (the only Eigen user)

`method="homotopy"` is a second, older solver kept as a fallback. It is the one
piece that needs Eigen, so it is **off by default** — enabling it is what pulls
in the dependency, and with it the network fetch:

```bash
pip install -C cmake.define.QSP_ANGLES_WITH_HOMOTOPY=ON .
```

With it on, a system Eigen is preferred (`apt install libeigen3-dev`); failing
that CMake fetches Eigen 3.4.0 from `gitlab.com/libeigen/eigen`, which requires
network access. To stay offline, point it at a local checkout:

```bash
pip install . \
  -C cmake.define.QSP_ANGLES_WITH_HOMOTOPY=ON \
  -C cmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN3=/path/to/eigen
```

Check what you have with `qsp_angles.HAS_HOMOTOPY`; asking for a solver the build
lacks raises a `ValueError` naming the flag. The default `sym_qsp` is both faster
and higher-degree, so leaving this off costs nothing in normal use.

### CI and wheels

- Continuous integration lives at the repo root:
  `.github/workflows/qsp-angles-ci.yml` (path-scoped to `packages/qsp-angles/**`),
  in three jobs:
  - **`test`** — the default build, on a runner with **no Eigen installed**
    (it asserts `/usr/include/eigen3` is absent first). Runs the Python tests,
    the stdlib-only C++ smoke test, cross-validation against `pyqsp`, the pure-C
    ABI client, the ctypes client, the Rust example, and the vendored-source
    sync check. If a third-party dependency ever creeps back into the core,
    this job goes red.
  - **`abi-no-third-party-deps`** — compiles the C ABI in an empty directory
    with no `-I` flags at all under `-Werror`, then links it from pure C.
  - **`homotopy-optional`** — the opt-in Eigen path, which also asserts the two
    `response` implementations agree bit-for-bit.
- Wheels are built by the repo-root workflow
  `.github/workflows/qsp-angles-wheels.yml` (triggered on `qsp-angles-v*` tags).

## License

MIT.
