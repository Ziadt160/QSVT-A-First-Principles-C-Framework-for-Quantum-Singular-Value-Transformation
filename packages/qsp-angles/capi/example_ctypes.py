"""Pure-stdlib Python client of the qsp-angles C ABI via ctypes.

This needs NO build of the package and NO python3-dev / pybind11 — just the
prebuilt shared library and Python's standard `ctypes`. It is the
"Python-without-pybind" integration path, and a quick way to use the solver from
Python on a box where the wheel can't build.

Build the shared library first (see capi/README.md). No third-party headers are
needed — the solver behind the ABI is stdlib-only C++17:

    g++ -O3 -fPIC -shared -std=c++17 -I../src \
        qsp_angles.cpp ../src/SymQspAngleSolver.cpp \
        -o libqsp_angles_c.so

    python example_ctypes.py ./libqsp_angles_c.so
"""

import ctypes
import math
import sys

D = ctypes.c_double
PD = ctypes.POINTER(D)
PI = ctypes.POINTER(ctypes.c_int)


def load(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.qsp_version.restype = ctypes.c_char_p
    lib.qsp_solve_chebyshev.restype = ctypes.c_int
    lib.qsp_solve_chebyshev.argtypes = [ctypes.c_int, PD, ctypes.c_int, PD, ctypes.c_int, PI, PD, PI]
    lib.qsp_response.restype = D
    lib.qsp_response.argtypes = [D, PD, ctypes.c_int]
    return lib


def solve_chebyshev(lib, degree, coeffs):
    n = degree + 1
    assert len(coeffs) == n
    c = (D * n)(*coeffs)
    phases = (D * n)()
    length = ctypes.c_int(0)
    residual = D(0.0)
    converged = ctypes.c_int(0)
    st = lib.qsp_solve_chebyshev(
        degree, c, n, phases, n, ctypes.byref(length),
        ctypes.byref(residual), ctypes.byref(converged),
    )
    if st != 0:
        raise RuntimeError(f"qsp_solve_chebyshev failed with status {st}")
    return list(phases[: length.value]), residual.value, bool(converged.value)


def main():
    lib_path = sys.argv[1] if len(sys.argv) > 1 else "./libqsp_angles_c.so"
    lib = load(lib_path)
    print("qsp-angles", lib.qsp_version().decode(), "(ctypes client)")

    # 0.8 * T_5(x): full Chebyshev coefficients c_0..c_5.
    phases, residual, converged = solve_chebyshev(lib, 5, [0, 0, 0, 0, 0, 0.8])
    print(f"degree 5 -> {len(phases)} phases, residual = {residual:.2e}, converged = {converged}")

    arr = (D * len(phases))(*phases)
    max_err = 0.0
    for i in range(21):
        x = -1.0 + 2.0 * i / 20.0
        got = lib.qsp_response(x, arr, len(phases))
        want = 0.8 * math.cos(5.0 * math.acos(max(-1.0, min(1.0, x))))
        max_err = max(max_err, abs(got - want))
    print(f"max |Re<0|U|0> - 0.8*T_5| over grid = {max_err:.2e}")
    sys.exit(0 if max_err < 1e-10 else 2)


if __name__ == "__main__":
    main()
