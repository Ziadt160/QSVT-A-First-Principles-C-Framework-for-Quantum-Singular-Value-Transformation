# qsp-angles

**A fast, embeddable QSP/QSVT phase-factor (angle) solver — one `pip install`, a `pyqsp`-shaped API, a C++ core.**

Quantum Signal Processing and Quantum Singular Value Transformation both reduce
to one classical pre-computation: given a target polynomial `f(x)`, find the
phase sequence `Φ` that makes a QSP circuit realise `f`. This package does *only*
that, fast, with no quantum-simulator dependency to install.

```bash
pip install qsp-angles
```

```python
import qsp_angles as qa

# From a callable target (|f| <= 1 on [-1,1], parity = degree mod 2):
r = qa.target2angles(lambda x: 0.7 * x, degree=1)
r.phases       # -> numpy array of phase factors (Wx convention)
r.residual     # -> worst-case error vs the target on [-1, 1]
r.converged    # -> bool

# From polynomial coefficients (monomial basis, ascending):
qa.poly2angles([0.0, 0.5])           # 0.5 * x

# Evaluate the achieved QSP polynomial:
qa.response(0.3, r.phases)
```

### Drop-in for pyqsp

If you already call `pyqsp`, swap the import for the common entry point:

```python
# from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from qsp_angles import QuantumSignalProcessingPhases

phases = QuantumSignalProcessingPhases(poly, signal_operator="Wx")
```

Same **Wx** convention pyqsp defaults to. This is API-compatible for the common
call shape — it returns the symmetric phase sequence in the Wx convention; it is
not guaranteed bit-identical to pyqsp's output. Only `Wx` is supported today.

## Why this exists

`pyqsp` (Python) and `QSPPACK` (MATLAB) are the established angle solvers — but
both are interpreted and not embeddable in a compiled stack. This is, to our
knowledge, the **first standalone C++ QSP angle solver**, packaged so Python
users get it transparently. It is extracted from a full from-scratch C++ QSVT
framework ([repo](https://github.com/Ziadt160/QSVT-A-First-Principles-C-Framework-for-Quantum-Singular-Value-Transformation)).

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

## License

MIT.
