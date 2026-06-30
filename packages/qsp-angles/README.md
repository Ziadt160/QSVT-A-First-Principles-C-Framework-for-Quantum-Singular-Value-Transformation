# qsp-angles

**An embeddable C++ QSP/QSVT phase-factor (angle) solver — no Python required at
the core, usable from C++/compiled stacks, with a thin `pip install` Python
binding on top.**

Quantum Signal Processing and Quantum Singular Value Transformation both reduce
to one classical pre-computation: given a target polynomial `f(x)`, find the
phase sequence `Φ` that makes a QSP circuit realise `f`. This package does *only*
that — and it does it in a **self-contained C++ core** (Eigen + STL, no
quantum-simulator dependency) that you can drop straight into a compiled
codebase. The Python wheel is a convenience on top, not the substrate.

It matches the established interpreted solvers on **accuracy** (machine precision)
up to about **degree 100**, and is **speed-competitive at low degree** (≲20–30);
beyond that, `pyqsp`'s Newton solver is faster (~3–4× at degree 101) and reaches
far higher degree. Its differentiator is **embeddability**, not raw speed — see
[bench/BENCHMARK.md](bench/BENCHMARK.md) for the head-to-head numbers.

```bash
pip install qsp-angles
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
are interpreted and not embeddable in a compiled stack. This package's value
proposition is the **embeddable C++ core**: the solver is two C++ files (Eigen +
STL) with no Python in the loop, so you can call it directly from C++ or any
compiled/native pipeline, and link it without dragging in a Python runtime. The
pip wheel is a thin pybind11 binding for the Python crowd. It is, to our
knowledge, the first standalone C++ QSP angle solver. It is extracted from a full
from-scratch C++ QSVT framework
([repo](https://github.com/Ziadt160/QSVT-A-First-Principles-C-Framework-for-Quantum-Singular-Value-Transformation)).

This is not a claim of being "better than pyqsp." It matches pyqsp on accuracy
up to ~degree 100 and is speed-competitive only at low degree; pyqsp is faster at
high degree and reaches degree > 1000. Its differentiator is embeddability, not
speed or degree reach — see [bench/BENCHMARK.md](bench/BENCHMARK.md).

The core combines two ideas that let plain C++/Eigen reach high degree reliably:

- **Homotopy continuation** — morph the target from `T_d` (which `Φ = 0` solves
  exactly) to `f`, warm-starting each step, so every sub-problem stays near a
  known optimum.
- **Exact analytic Jacobian** — differentiate the QSP product in closed form via
  prefix/suffix products (`O(d)` per node), ~10× faster than finite differences
  at degree 100.

| target               | degree | residual | time (`-O3 -march=native`) |
|----------------------|--------|----------|----------------------------|
| `0.7 T_31`           | 31     | ~5e-15   | ~50 ms                     |
| `0.8 T_71`           | 71     | ~1e-14   | ~480 ms                    |
| `0.7 T_101`          | 101    | ~1e-14   | ~1.7 s                     |

## Honest limits

- Reaches degree ~100+ at machine precision; it does **not** yet hit the
  `degree > 1000` regime of `sym_qsp`/root-finding solvers. The principled next
  step is a non-iterative Fejér–Riesz completion (machine-precision by
  construction).
- Supports the `Wx` convention only.

## Building from source

```bash
pip install .            # needs a C++17 compiler; Eigen is auto-fetched if absent
pip install -e .[test]   # editable + test deps
pytest tests
```

The wheel vendors only two C++ files (`src/QspAngleSolver.{hpp,cpp}`) plus a thin
pybind11 layer. No Qrack, no CUDA, no system Eigen required.

### Eigen dependency (network requirement / override)

Eigen is header-only. The build prefers a **system Eigen** if `find_package`
locates one (fast, offline). Otherwise CMake **fetches Eigen 3.4.0 over the
network** via `FetchContent` from `https://gitlab.com/libeigen/eigen.git` —
which means an isolated/offline build with no system Eigen will fail at
configure time. The configure step prints a clear status line saying whether it
found a system Eigen or is fetching.

To build offline (or to pin a local checkout), point CMake at an Eigen source
tree and it will skip the network fetch:

```bash
# Use a local Eigen checkout instead of fetching:
pip install . --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN3=/path/to/eigen

# Or rely on a system install (e.g. apt install libeigen3-dev) — auto-detected.
```

### CI and wheels

- Continuous integration lives at the repo root:
  `.github/workflows/qsp-angles-ci.yml` (path-scoped to `packages/qsp-angles/**`)
  builds + tests the package, runs the C++ smoke test, and checks the vendored
  sources are in sync with the repo root.
- Wheels are built by the repo-root workflow
  `.github/workflows/qsp-angles-wheels.yml` (triggered on `qsp-angles-v*` tags).

## License

MIT.
