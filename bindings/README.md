# Python bindings

`qsvt_native` exposes the C++ QSVT toolkit to Python via pybind11. Eigen
matrices map to/from NumPy arrays automatically, and Python callables can be
passed as polynomial targets.

## API

```python
import numpy as np
import qsvt_native as q

# QSP angle finding for a target polynomial (degree d, parity d mod 2, |f|<=1)
res = q.poly_to_angles(lambda x: 0.7 * x, 1)      # -> {phases, residual, converged}
q.qsp_response(0.3, res["phases"])                 # achieved polynomial at x

# Unitary -> native gates, with OpenQASM + resource estimate
d = q.shannon_decompose(U)        # n-qubit unitary  -> {cnot, depth, qasm, ...}
d = q.two_qubit_synthesize(U4)    # 4x4 unitary

# Full QSVT
q.rotation_block_encoding(A)                       # U_W of Hermitian A
q.qsvt_block(A, phases)                            # P(A)
prog = q.compile_matrix_function(A, f, degree)     # -> {phases, cnot, qasm, ...}

# Applications
q.hamiltonian_evolution(H, t, degree)              # ~ e^{-iHt}  (dense)
q.spectral_projector(H, mu=0.0, w=0.1, degree=25)  # ~ projector onto lambda > mu
q.lcu_coefficients(A)                              # Pauli decomposition coeffs
```

See `smoke_test.py` for a runnable end-to-end example.

## Building

**Via CMake (recommended; needs `python3-dev` + `pip install pybind11`):**

```bash
cmake -S . -B build -DBUILD_PYTHON=ON
cmake --build build --target qsvt_native
# the qsvt_native*.so lands under build/bindings/
```

**Standalone (no CMake), if you only have the headers and pybind11:**

```bash
g++ -O3 -march=native -shared -fPIC -std=c++17 bindings/qsvt_py.cpp \
  -I"$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')" \
  -I"$(python3 -m pybind11 --includes | sed 's/-I//;s/ .*//')" \
  -I/usr/include/eigen3 -Iinclude -I/usr/local/include \
  -Lbuild/src -lqsvt_lib \
  -Wl,--start-group /usr/local/lib/qrack/libqrack.a -lcudart -Wl,--end-group \
  -lpthread -o qsvt_native.so
```

A Python extension module does not link `libpython` itself; the interpreter
provides those symbols at import time.

## Test

```bash
PYTHONPATH=. python3 bindings/smoke_test.py
```
