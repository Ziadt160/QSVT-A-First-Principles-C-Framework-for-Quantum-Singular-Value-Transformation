# qsp-angles C ABI

A flat `extern "C"` interface ([qsp_angles.h](qsp_angles.h)) so the solver can be
embedded in any compiled toolchain — C, Rust, Julia, Go, or C++ — with **no
Python runtime** and no C++ ABI concerns. This is the package's reason to exist:
the established solvers (pyqsp, QSPPACK) are interpreted; this one you can link.

## The API

```c
#include "qsp_angles.h"

int    qsp_num_phases(int degree);                    /* degree + 1 */
qsp_status qsp_solve_chebyshev(int degree,
                 const double* cheb_coeffs, int ncoeffs,   /* c_0..c_d of f */
                 double* out_phases, int out_cap, int* out_len,
                 double* out_residual, int* out_converged);
qsp_status qsp_solve_callback(int degree, qsp_target_fn f, void* ctx, ...);
double qsp_response(double x, const double* phases, int len);  /* Re<0|U|0> */
const char* qsp_version(void);
const char* qsp_status_string(qsp_status);
```

Phases are the `degree+1` full symmetric phases in the **Wx convention**
(`Re<0|U(x)|0> = f(x)`). The target must satisfy `|f| <= 1` and parity
`degree % 2`. Exceptions never cross the boundary — failures are status codes.

## Build

### CMake (static + shared lib + the example)

```bash
cd capi && cmake -S . -B build && cmake --build build
ctest --test-dir build        # runs the pure-C example
```

### By hand (shows the C-links-C++ story)

```bash
# Compile the C++ side (impl + solver) once:
g++ -O3 -std=c++17 -I../src -I/usr/include/eigen3 -c \
    qsp_angles.cpp ../src/SymQspAngleSolver.cpp ../src/QspAngleSolver.cpp
ar rcs libqsp_angles_c.a qsp_angles.o SymQspAngleSolver.o QspAngleSolver.o

# Compile + link a PURE C client (note: gcc, not g++):
gcc -O2 example_c.c -L. -lqsp_angles_c -lstdc++ -lm -o example_c
./example_c
```

`-lstdc++` is required because the implementation is C++ — but your code stays
pure C. (For a fully C-runtime build you would compile the impl with a C++
compiler that statically links libstdc++; the ABI itself is plain C.)

## Binding from other languages

- **Rust:** declare the externs (or generate with `bindgen`) and link the static
  lib via a `build.rs`; wrap `qsp_solve_chebyshev` in a safe `fn`. A thin
  `qsp-angles` crate is the natural next step.
- **Julia:** `ccall((:qsp_solve_chebyshev, "libqsp_angles_c"), ...)`.
- **Go:** cgo with `#include "qsp_angles.h"`.
- **Python without pybind:** `ctypes`/`cffi` against the shared lib — see
  [example_ctypes.py](example_ctypes.py) (stdlib only; works with no `python3-dev`
  and no wheel build). The pip wheel uses pybind11; this is the dependency-free
  alternative.
- **Rust:** a thin safe crate lives in [../rust/](../rust) (`build.rs` compiles
  this C ABI via the `cc` crate).

## Status codes

`QSP_OK`, `QSP_ERR_NULL`, `QSP_ERR_DEGREE`, `QSP_ERR_NCOEFFS`, `QSP_ERR_BUFFER`,
`QSP_ERR_NOT_CONVERGED`, `QSP_ERR_INTERNAL` — use `qsp_status_string()` for names.
